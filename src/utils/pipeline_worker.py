from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from mapping.video_extractor import VideoExtractor
from mapping.stitcher import SimpleStitcher
from mapping.postprocess import crop_black, crop_largest_inner_rect


def next_available_path(base_path: str) -> str:
    """
    Returns base_path if it doesn't exist, otherwise base_path with an
    incrementing suffix:  stitched_map.png → stitched_map1.png → stitched_map2.png

    Note: There is an inherent TOCTOU window between checking existence and
    writing.  For a single-user desktop app this is acceptable; the write call
    in SaveWorker is wrapped in a try/except as an additional safety net.
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

    save_ok = Signal(str)   # path that was written
    save_err = Signal(str)  # error message

    def __init__(self, pano: np.ndarray, base_path: str, parent=None):
        super().__init__(parent)
        # Keep our own reference; the caller may release theirs
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
            # Release the copy regardless of outcome
            self._pano = None


class PipelineWorker(QThread):
    progress    = Signal(float, str)   # value 0..1, message
    finished_ok = Signal(object)       # pano (np.ndarray)
    finished_err = Signal(str)         # error message

    def __init__(
        self,
        video_path: str,
        seconds_step: float,
        max_frames: int,
        stitch_mode: str = "scans",
        work_megapix: float = 1.5,
        min_keypoints: int = 120,
        orb_nfeatures: int = 2000,
        extract_megapix: float = 2.0,
        similar_threshold: float = 6.0,
    ):
        super().__init__()
        self.video_path = str(video_path)
        self.seconds_step = float(seconds_step)
        self.max_frames = int(max_frames)

        self.stitch_mode = str(stitch_mode)
        self.work_megapix = float(work_megapix)
        self.min_keypoints = int(min_keypoints)
        self.orb_nfeatures = int(orb_nfeatures)

        self.extract_megapix = float(extract_megapix)
        self.similar_threshold = float(similar_threshold)

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
            frames = extractor.extract(on_progress=lambda p, m: cb(p * 0.50, m))
            self._check_cancel()

            stitcher = SimpleStitcher(
                mode=self.stitch_mode,
                work_megapix=self.work_megapix,
                min_keypoints=self.min_keypoints,
                orb_nfeatures=self.orb_nfeatures,
            )
            pano = stitcher.stitch(
                frames,
                on_progress=lambda p, m: cb(0.50 + p * 0.45, m),
                cancel_check=self._check_cancel,
            )

            # Release frames as soon as stitching is done
            frames = None
            self._check_cancel()

            cb(0.95, "Post-processing map...")
            pano = crop_black(pano)
            pano = crop_largest_inner_rect(pano, shrink_iters=3)

            self._check_cancel()
            cb(1.0, "Done")
            self.finished_ok.emit(pano)

        except Exception as e:
            frames = None
            pano = None

            msg = str(e).strip() or e.__class__.__name__
            if e.__class__.__name__ not in msg:
                msg = f"{e.__class__.__name__}: {msg}"

            self.finished_err.emit(msg)