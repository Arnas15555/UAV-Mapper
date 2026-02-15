from __future__ import annotations
import os
import cv2
from PySide6.QtCore import QThread, Signal

from mapping.video_extractor import VideoExtractor
from mapping.stitcher import SimpleStitcher
from mapping.postprocess import crop_black, auto_rotate
from mapping.postprocess import crop_largest_inner_rect

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
    progress = Signal(float, str)   # value, message
    finished_ok = Signal(object)    # pano (np.ndarray)
    finished_err = Signal(str)

    def __init__(self, video_path: str, seconds_step: float, max_frames: int):
        super().__init__()
        self.video_path = video_path
        self.seconds_step = seconds_step
        self.max_frames = max_frames

    def run(self):
        try:
            def cb(p, msg):
                self.progress.emit(float(p), str(msg))

            cb(0.0, "Starting...")

            extractor = VideoExtractor(
                self.video_path,
                seconds_step=self.seconds_step,
                max_frames=self.max_frames
            )
            frames = extractor.extract(on_progress=lambda p, m: cb(p * 0.5, m))  # 0–50%

            stitcher = SimpleStitcher(mode="scans", work_megapix=1.5, min_keypoints=120)
            pano = stitcher.stitch(frames, on_progress=lambda p, m: cb(0.5 + p * 0.45, m))  # 50–95%

            cb(0.95, "Post-processing map...")
            pano = crop_black(pano)
            #pano = auto_rotate(pano)
            #pano = crop_black(pano)
            pano = crop_largest_inner_rect(pano, shrink_iters=3)
            #pano = cv2.resize(pano, (3840, 2160), interpolation=cv2.INTER_AREA)

            cb(1.0, "Done")
            self.finished_ok.emit(pano)

        except Exception as e:
            self.finished_err.emit(str(e))
