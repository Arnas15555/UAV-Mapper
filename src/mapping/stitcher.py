from __future__ import annotations

import cv2
import numpy as np
from typing import Callable, List, Optional, Tuple

from utils.config import DEFAULTS

ProgressCb = Optional[Callable[[float, str], None]]


class SimpleStitcher:
    """
    Smarter stitcher:
    - Downscale
    - Compute ORB once per frame
    - Filter low-feature frames
    - Greedy frame selection using a balanced score:
        score = inliers - motion_penalty
      where motion_penalty prefers motion near a target (avoids "best overlap only")
    - Overlap quality from homography RANSAC inliers
    - Motion heuristic from median displacement of inlier correspondences (robust)
    - KNN matching + Lowe ratio test (more robust than BF crossCheck)
    - Cancel checks inside long loops
    - Duplicate-frame guard in greedy selection
    """

    def __init__(
        self,
        mode: str = DEFAULTS["stitch_mode"],
        work_megapix: float = DEFAULTS["work_megapix"],
        min_keypoints: int = DEFAULTS["min_keypoints"],
        orb_nfeatures: int = DEFAULTS["orb_nfeatures"],
        lookahead: int = 6,
        min_inliers: int = 30,
        min_motion_px: float = 8.0,
        target_motion_px: float = 40.0,
        motion_weight: float = 0.35,
        ratio_thresh: float = 0.75,
        ransac_reproj_thresh: float = 4.0,
        max_frames_for_stitch: int = 60,
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

        self._orb = cv2.ORB_create(nfeatures=self.orb_nfeatures)
        # KNN matching needs crossCheck=False
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

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
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _compute_features(
        self, img_bgr: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kps, des = self._orb.detectAndCompute(gray, None)
        return kps, des

    def _match_knn_ratio(
        self, des_a: np.ndarray, des_b: np.ndarray
    ) -> List[cv2.DMatch]:
        """KNN + Lowe ratio test. Returns best matches sorted by distance."""
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
        """
        Returns (inliers_count, median_inlier_displacement_px).
        Returns (0, 0.0) if quality cannot be estimated.
        """
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

        # Median displacement of inlier correspondences (robust to outliers)
        da = pts_a[mask]
        db = pts_b[mask]
        disp = np.linalg.norm(db - da, axis=1)
        med_disp = float(np.median(disp)) if disp.size else 0.0

        return inliers, med_disp

    def _score_candidate(self, inliers: int, motion: float) -> float:
        """
        Balanced score:
        - primary: maximize inliers
        - secondary: prefer motion near target_motion_px
        """
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

        selected_idxs: List[int] = [usable[0]]
        # Track selected set for fast duplicate checking
        selected_set: set[int] = {usable[0]}
        cur_pos = 0

        # Greedy selection with lookahead
        while len(selected_idxs) < self.max_frames_for_stitch and cur_pos < len(usable) - 1:
            if cancel_check:
                cancel_check()

            base_idx = usable[cur_pos]
            kps_a, des_a = feats[base_idx]

            best = None  # (score, inliers, motion, next_pos, cand_idx)
            max_pos = min(len(usable) - 1, cur_pos + self.lookahead)

            for next_pos in range(cur_pos + 1, max_pos + 1):
                if cancel_check:
                    cancel_check()

                cand_idx = usable[next_pos]

                # Skip already-selected frames to avoid duplicates
                if cand_idx in selected_set:
                    continue

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
                # Fallback: advance one step; find the next usable frame not yet selected
                next_pos = cur_pos + 1
                while next_pos < len(usable) and usable[next_pos] in selected_set:
                    next_pos += 1
                if next_pos >= len(usable):
                    break
                cur_pos = next_pos
                chosen_idx = usable[cur_pos]
            else:
                _, _, _, cur_pos, chosen_idx = best

            selected_idxs.append(chosen_idx)
            selected_set.add(chosen_idx)

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

        cb(0.02, f"Preparing {len(frames)} frames...")

        # Downscale into a separate list — avoids doubling RAM by not modifying originals
        frames_small = [self._downscale_to_megapix(f, self.work_megapix) for f in frames]
        cb(0.08, "Downscaled frames")

        selected = self._select_frames(frames_small, on_progress=on_progress, cancel_check=cancel_check)

        # Free the full downscaled list as soon as selection is done
        del frames_small

        if len(selected) < 2:
            raise RuntimeError("Frame selection produced <2 frames. Try different settings.")

        stitch_mode = cv2.Stitcher_SCANS if self.mode == "scans" else cv2.Stitcher_PANORAMA
        stitcher = cv2.Stitcher_create(stitch_mode)

        try:
            stitcher.setPanoConfidenceThresh(0.5)
        except Exception:
            pass

        if cancel_check:
            cancel_check()

        cb(0.85, "Stitching selected frames...")
        status, pano = stitcher.stitch(selected)

        # Free selected frames immediately after stitching
        del selected

        if status != cv2.Stitcher_OK:
            raise RuntimeError(
                f"Stitching failed (status={status}). "
                f"Try: fewer max frames, different mode, smaller seconds step, more overlap."
            )

        cb(1.0, "Stitch complete")
        return pano