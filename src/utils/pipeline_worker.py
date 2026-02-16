from __future__ import annotations

import os
from typing import Optional

import cv2
from PySide6.QtCore import QThread, Signal

from mapping.video_extractor import VideoExtractor
from mapping.stitcher import SimpleStitcher
from mapping.postprocess import crop_black, crop_largest_inner_rect


def next_available_path(base_path: str) -> str:
    """
    If base_path exists, returns base1, base2, ...
    Example: stitched_map.png -> stitched_map1.png -> stitched_map2.png
    """
    if not os.path.exists(base_path):
        return base_path

    folder = os.path.dirname(base_path)
    name = os.path.basename(base_path)
    stem, ext = os.path.splitext(name)

    i = 1
    while True:
        candidate = os.path.join(folder, f"{stem}{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


class PipelineWorker(QThread):
    progress = Signal(float, str)   # value 0..1, message
    finished_ok = Signal(object)    # pano (np.ndarray)
    finished_err = Signal(str)      # error message

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
                # clamp p for safety
                p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else float(p))
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
            # Include type for debugging without dumping a full traceback into the UI
            if e.__class__.__name__ not in msg:
                msg = f"{e.__class__.__name__}: {msg}"

            self.finished_err.emit(msg)
