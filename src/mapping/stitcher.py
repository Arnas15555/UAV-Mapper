# src/mapping/stitcher.py
from __future__ import annotations

import cv2
from typing import Callable, List, Optional


ProgressCb = Optional[Callable[[float, str], None]]


class SimpleStitcher:

    def __init__(
        self,
        mode: str = "scans",
        work_megapix: float = 1.2,
        min_keypoints: int = 250,
        orb_nfeatures: int = 2000,
    ):
        self.mode = mode.lower().strip()
        self.work_megapix = float(work_megapix)
        self.min_keypoints = int(min_keypoints)
        self.orb_nfeatures = int(orb_nfeatures)

    @staticmethod
    def _downscale_to_megapix(img, mp: float):

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

    def _count_orb_keypoints(self, img) -> int:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=self.orb_nfeatures)
        kps = orb.detect(gray, None)
        return len(kps)

    def stitch(self, frames: List, on_progress: ProgressCb = None):
        if len(frames) < 2:
            raise RuntimeError("Need at least 2 frames to stitch.")

        def cb(p: float, msg: str):
            if on_progress:
                on_progress(max(0.0, min(1.0, p)), msg)

        cb(0.05, f"Preparing {len(frames)} frames...")

        small = [self._downscale_to_megapix(f, self.work_megapix) for f in frames]

        cb(0.15, "Filtering low-feature frames...")
        filtered = []
        kp_counts = []

        for i, img in enumerate(small):
            kp = self._count_orb_keypoints(img)
            kp_counts.append(kp)
            if kp >= self.min_keypoints:
                filtered.append(img)

        if len(filtered) < 2:

            best = max(kp_counts) if kp_counts else 0
            raise RuntimeError(
                f"Not enough usable frames for stitching. "
                f"Frames kept: {len(filtered)}/{len(frames)}. "
                f"Best keypoints: {best}. "
                f"Try slower movement, better lighting, less blur, or reduce min_keypoints."
            )

        cb(0.25, f"Kept {len(filtered)}/{len(frames)} frames (>= {self.min_keypoints} keypoints).")

        stitch_mode = cv2.Stitcher_SCANS if self.mode == "scans" else cv2.Stitcher_PANORAMA
        stitcher = cv2.Stitcher_create(stitch_mode)

        try:
            stitcher.setPanoConfidenceThresh(0.5)
        except Exception:
            pass

        cb(0.35, "Stitching frames (OpenCV Stitcher)...")
        status, pano = stitcher.stitch(filtered)

        if status != cv2.Stitcher_OK:
            raise RuntimeError(
                f"Stitching failed (status={status}). "
                f"Try: fewer frames, smaller seconds_step, more overlap, or steadier motion."
            )

        cb(1.0, "Stitch complete")
        return pano
