from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from mapping.video_extractor import VideoExtractor
from mapping.stitcher import SimpleStitcher
from mapping.postprocess import crop_black, crop_largest_inner_rect, auto_rotate


def next_available_path(base_path: str) -> str:
    """
    Returns base_path if it doesn't exist, otherwise base_path with an
    incrementing suffix:  stitched_map.png → stitched_map1.png → stitched_map2.png
    """
    if not os.path.exists(base_path):
        return base_path

    folder = os.path.dirname(base_path) or "."
    name = os.path.basename(base_path)
    stem, ext = os.path.splitext(name)

    i = 1
    while True:
        candidate = os.path.join(folder, f"{stem}{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


class SaveWorker(QThread):
    """
    Saves a numpy array (BGR image) to disk on a background thread so the
    main thread — and therefore the UI — is not blocked by a large imwrite.
    """

    save_ok  = Signal(str)   # path that was written
    save_err = Signal(str)   # error message

    def __init__(self, pano: np.ndarray, base_path: str, parent=None):
        super().__init__(parent)
        self._pano = pano.copy()
        self._base_path = base_path

    def run(self):
        try:
            out_path = next_available_path(self._base_path)
            ok = cv2.imwrite(out_path, self._pano)
            if not ok:
                self.save_err.emit(f"cv2.imwrite returned False for path: {out_path}")
                return
            self.save_ok.emit(out_path)
        except Exception as e:
            self.save_err.emit(f"{e.__class__.__name__}: {e}")
        finally:
            self._pano = None


class PipelineWorker(QThread):
    progress     = Signal(float, str)   # value 0..1, message
    finished_ok  = Signal(object)       # pano (np.ndarray)
    finished_err = Signal(str)          # error message

    def __init__(
        self,
        video_path: str,
        seconds_step: float,
        max_frames: int,
        stitch_mode: str = "panorama",       # FIX: panorama mode suits UAV footage better
        work_megapix: float = 2.0,
        min_keypoints: int = 100,
        orb_nfeatures: int = 4000,
        extract_megapix: float = 2.0,
        similar_threshold: float = 8.0,
        min_motion_px: float = 3.0,          # FIX: was 8.0 — too strict for slow UAV
        target_motion_px: float = 20.0,      # FIX: was 40.0 — more realistic for UAV
        max_frames_for_stitch: int = 120,    # FIX: was hard-coded 60
    ):
        super().__init__()
        self.video_path        = str(video_path)
        self.seconds_step      = float(seconds_step)
        self.max_frames        = int(max_frames)
        self.stitch_mode       = str(stitch_mode)
        self.work_megapix      = float(work_megapix)
        self.min_keypoints     = int(min_keypoints)
        self.orb_nfeatures     = int(orb_nfeatures)
        self.extract_megapix   = float(extract_megapix)
        self.similar_threshold = float(similar_threshold)
        self.min_motion_px     = float(min_motion_px)
        self.target_motion_px  = float(target_motion_px)
        self.max_frames_for_stitch = int(max_frames_for_stitch)

    def _check_cancel(self):
        if self.isInterruptionRequested():
            raise RuntimeError("Cancelled")

    def run(self):
        frames: Optional[list] = None
        pano = None

        try:
            def cb(p: float, msg: str):
                p = max(0.0, min(1.0, float(p)))
                self.progress.emit(p, str(msg))

            cb(0.0, "Starting...")
            self._check_cancel()

            extractor = VideoExtractor(
                self.video_path,
                seconds_step=self.seconds_step,
                max_frames=self.max_frames,
                extract_megapix=self.extract_megapix,
                similar_threshold=self.similar_threshold,
                cancel_check=self._check_cancel,
            )
            frames = extractor.extract(on_progress=lambda p, m: cb(p * 0.45, m))
            self._check_cancel()

            stitcher = SimpleStitcher(
                mode=self.stitch_mode,
                work_megapix=self.work_megapix,
                min_keypoints=self.min_keypoints,
                orb_nfeatures=self.orb_nfeatures,
                min_motion_px=self.min_motion_px,
                target_motion_px=self.target_motion_px,
                max_frames_for_stitch=self.max_frames_for_stitch,
            )
            pano = stitcher.stitch(
                frames,
                on_progress=lambda p, m: cb(0.45 + p * 0.45, m),
                cancel_check=self._check_cancel,
            )

            frames = None
            self._check_cancel()

            cb(0.92, "Post-processing map...")
            pano = crop_black(pano)

            # FIX: auto_rotate was defined in postprocess.py but never called.
            # Corrects the ~45° tilt that accumulates from homography chaining.
            cb(0.95, "Auto-rotating map...")
            pano = auto_rotate(pano)

            cb(0.97, "Cropping to content...")
            pano = crop_largest_inner_rect(pano, shrink_iters=3)

            self._check_cancel()
            cb(1.0, "Done")
            self.finished_ok.emit(pano)

        except Exception as e:
            frames = None
            pano   = None

            msg = str(e).strip() or e.__class__.__name__
            if e.__class__.__name__ not in msg:
                msg = f"{e.__class__.__name__}: {msg}"

            self.finished_err.emit(msg)