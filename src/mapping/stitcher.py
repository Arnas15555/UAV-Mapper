from __future__ import annotations

import cv2
import numpy as np
from typing import Callable, List, Optional, Tuple

ProgressCb = Optional[Callable[[float, str], None]]

# ---------------------------------------------------------------------------
# CUDA detection — falls back to CPU silently if not available
# ---------------------------------------------------------------------------
_CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0


class SimpleStitcher:
    """
    Stitches a sequence of UAV frames into a single panorama.

    Frame selection uses a greedy algorithm scored as:
        score = inliers - motion_weight * abs(motion - target_motion_px)

    This rewards good overlap while penalising motion that deviates from the
    target, avoiding the failure mode of always picking the single most-overlapping
    pair at the expense of spatial coverage.

    Overlap quality comes from RANSAC homography inlier counts.
    Motion is the median displacement of inlier correspondences (robust to outliers).
    Matching uses KNN + Lowe ratio test rather than brute-force cross-check.

    CUDA is used automatically when an NVIDIA GPU is available.
    """

    def __init__(
        self,
        mode: str = "panorama",
        work_megapix: float = 2.0,
        min_keypoints: int = 100,
        orb_nfeatures: int = 4000,
        lookahead: int = 6,
        min_inliers: int = 20,
        min_motion_px: float = 3.0,
        target_motion_px: float = 20.0,
        motion_weight: float = 0.35,
        ratio_thresh: float = 0.75,
        ransac_reproj_thresh: float = 4.0,
        max_frames_for_stitch: int = 120,
        max_match_keep: int = 400,
    ):
        self.mode = mode.lower().strip()
        self.work_megapix = float(work_megapix)
        self.min_keypoints = int(min_keypoints)
        self.orb_nfeatures = int(orb_nfeatures)

        self.lookahead = int(max(1, lookahead))
        self.min_inliers = int(max(0, min_inliers))
        self.min_motion_px = float(max(0.0, min_motion_px))
        self.target_motion_px = float(max(0.0, target_motion_px))
        self.motion_weight = float(max(0.0, motion_weight))

        self.ratio_thresh = float(ratio_thresh)
        self.ransac_reproj_thresh = float(ransac_reproj_thresh)
        self.max_frames_for_stitch = int(max(2, max_frames_for_stitch))
        self.max_match_keep = int(max(50, max_match_keep))

        if _CUDA_AVAILABLE:
            self._orb = cv2.cuda.ORB_create(nfeatures=self.orb_nfeatures)
            self._bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        else:
            self._orb = cv2.ORB_create(nfeatures=self.orb_nfeatures)
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # CLAHE instance reused across all frames (avoids repeated allocation)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _normalize_exposure(self, img: np.ndarray) -> np.ndarray:
        """
        Equalise per-frame exposure using CLAHE on the L channel of LAB colour space.
        This reduces the visible seam lines caused by exposure differences between frames
        without affecting hue or saturation.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _downscale_to_megapix(img: np.ndarray, mp: float) -> np.ndarray:
        if mp <= 0:
            return img
        h, w = img.shape[:2]
        cur_mp = (w * h) / 1_000_000.0
        if cur_mp <= mp:
            return img
        scale = (mp / cur_mp) ** 0.5
        new_w = max(64, int(w * scale))
        new_h = max(64, int(h * scale))

        if _CUDA_AVAILABLE:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_resized = cv2.cuda.resize(gpu_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return gpu_resized.download()
        else:
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _compute_features(
        self, img_bgr: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if _CUDA_AVAILABLE:
            gpu_gray = cv2.cuda_GpuMat()
            gpu_gray.upload(gray)
            gpu_kps, gpu_des = self._orb.detectAndComputeAsync(gpu_gray, None)
            kps = self._orb.convert(gpu_kps)
            des = gpu_des.download() if gpu_des is not None else None
        else:
            kps, des = self._orb.detectAndCompute(gray, None)

        return kps, des

    def _match_knn_ratio(
        self, des_a: np.ndarray, des_b: np.ndarray
    ) -> List[cv2.DMatch]:
        if _CUDA_AVAILABLE:
            gpu_des_a = cv2.cuda_GpuMat()
            gpu_des_b = cv2.cuda_GpuMat()
            gpu_des_a.upload(des_a)
            gpu_des_b.upload(des_b)
            knn = self._bf.knnMatch(gpu_des_a, gpu_des_b, k=2)
        else:
            knn = self._bf.knnMatch(des_a, des_b, k=2)

        good: List[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        good.sort(key=lambda m: m.distance)
        if len(good) > self.max_match_keep:
            good = good[: self.max_match_keep]
        return good

    def _pair_quality(
        self,
        kps_a: List[cv2.KeyPoint],
        des_a: Optional[np.ndarray],
        kps_b: List[cv2.KeyPoint],
        des_b: Optional[np.ndarray],
    ) -> Tuple[int, float]:
        if des_a is None or des_b is None:
            return 0, 0.0
        if len(kps_a) < 8 or len(kps_b) < 8:
            return 0, 0.0

        matches = self._match_knn_ratio(des_a, des_b)
        if len(matches) < 8:
            return 0, 0.0

        pts_a = np.float32([kps_a[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kps_b[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(
            pts_a, pts_b, cv2.RANSAC, self.ransac_reproj_thresh
        )
        if H is None or mask is None:
            return 0, 0.0

        mask = mask.ravel().astype(bool)
        inliers = int(mask.sum())
        if inliers < 4:
            return inliers, 0.0

        da = pts_a[mask]
        db = pts_b[mask]
        disp = np.linalg.norm(db - da, axis=1)
        med_disp = float(np.median(disp)) if disp.size else 0.0

        return inliers, med_disp

    def _score_candidate(self, inliers: int, motion: float) -> float:
        motion_penalty = self.motion_weight * abs(motion - self.target_motion_px)
        return float(inliers) - float(motion_penalty)

    def _select_frames(
        self,
        frames_small: List[np.ndarray],
        on_progress: ProgressCb,
        cancel_check=None,
    ) -> List[np.ndarray]:
        def cb(p: float, msg: str):
            if on_progress:
                on_progress(max(0.0, min(1.0, p)), msg)

        n = len(frames_small)

        cb(0.02, "Computing ORB features...")
        feats: List[Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]] = []
        kp_counts: List[int] = []

        for i, img in enumerate(frames_small):
            if cancel_check:
                cancel_check()
            kps, des = self._compute_features(img)
            feats.append((kps, des))
            kp_counts.append(len(kps))
            cb(0.02 + 0.18 * (i + 1) / max(1, n), f"Features {i+1}/{n}")

        usable = [i for i, c in enumerate(kp_counts) if c >= self.min_keypoints]
        if len(usable) < 2:
            best = max(kp_counts) if kp_counts else 0
            raise RuntimeError(
                f"Not enough usable frames. Kept {len(usable)}/{n}. "
                f"Best keypoints: {best}. Reduce min_keypoints or extract better frames."
            )

        cb(0.22, f"Usable frames: {len(usable)}/{n} (>= {self.min_keypoints} keypoints)")

        selected_idxs = [usable[0]]
        cur_pos = 0

        while len(selected_idxs) < self.max_frames_for_stitch and cur_pos < len(usable) - 1:
            if cancel_check:
                cancel_check()

            base_idx = usable[cur_pos]
            kps_a, des_a = feats[base_idx]

            best = None
            max_pos = min(len(usable) - 1, cur_pos + self.lookahead)

            for next_pos in range(cur_pos + 1, max_pos + 1):
                if cancel_check:
                    cancel_check()

                cand_idx = usable[next_pos]
                kps_b, des_b = feats[cand_idx]

                inliers, motion = self._pair_quality(kps_a, des_a, kps_b, des_b)

                if inliers < self.min_inliers:
                    continue
                if motion < self.min_motion_px:
                    continue

                score = self._score_candidate(inliers, motion)

                if best is None or score > best[0]:
                    best = (score, inliers, motion, next_pos, cand_idx)

            if best is None:
                cur_pos += 1
                # FIX: only append if index is valid, avoid duplicates
                next_idx = usable[cur_pos]
                if next_idx not in selected_idxs:
                    selected_idxs.append(next_idx)
            else:
                _, _, _, cur_pos, chosen_idx = best
                selected_idxs.append(chosen_idx)

            cb(
                0.22 + 0.58 * (len(selected_idxs) / max(2, min(self.max_frames_for_stitch, len(usable)))),
                f"Selecting frames... kept {len(selected_idxs)}"
            )

        out = [frames_small[i] for i in selected_idxs]
        cb(0.80, f"Selected {len(out)} frames for stitching")
        return out

    def stitch(self, frames: List[np.ndarray], on_progress: ProgressCb = None, cancel_check=None) -> np.ndarray:
        if len(frames) < 2:
            raise RuntimeError("Need at least 2 frames to stitch.")

        def cb(p: float, msg: str):
            if on_progress:
                on_progress(max(0.0, min(1.0, p)), msg)

        cb(0.02, f"Preparing {len(frames)} frames... ({'CUDA' if _CUDA_AVAILABLE else 'CPU'})")

        frames_small = [
            self._normalize_exposure(self._downscale_to_megapix(f, self.work_megapix))
            for f in frames
        ]
        cb(0.08, "Normalised exposure and downscaled frames")

        selected = self._select_frames(frames_small, on_progress=on_progress, cancel_check=cancel_check)
        if len(selected) < 2:
            raise RuntimeError("Frame selection produced <2 frames. Try different settings.")

        stitch_mode = cv2.Stitcher_SCANS if self.mode == "scans" else cv2.Stitcher_PANORAMA
        stitcher = cv2.Stitcher_create(stitch_mode)

        try:
            stitcher.setPanoConfidenceThresh(0.4)
        except Exception:
            pass

        try:
            # Multi-band blending produces smoother seams than the default feather blender,
            # especially where frames have slight exposure differences after normalisation.
            stitcher.setBlender(cv2.detail.MultiBandBlender())
        except Exception:
            pass

        if cancel_check:
            cancel_check()

        cb(0.85, f"Stitching {len(selected)} selected frames...")
        status, pano = stitcher.stitch(selected)

        if status != cv2.Stitcher_OK:
            raise RuntimeError(
                f"Stitching failed (status={status}). "
                f"Try: panorama mode, smaller seconds_step, more overlap between frames."
            )

        cb(1.0, "Stitch complete")
        return pano