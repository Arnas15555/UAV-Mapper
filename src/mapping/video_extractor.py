from __future__ import annotations

import os
from typing import Callable, Optional, List

import cv2
import numpy as np


class VideoExtractor:
    """
    Extract frames from a video using:
    - time-based sampling (seconds_step)
    - similarity filtering (skip near-duplicates)
    - downscaling at extraction time to reduce RAM/CPU usage
    - cancel_check hook for responsive cancellation
    """

    def __init__(
        self,
        video_path: str,
        seconds_step: float = 0.15,
        max_frames: int = 120,
        extract_megapix: float = 2.0,
        similar_threshold: float = 6.0,
        similar_resize: tuple[int, int] = (320, 180),
        similar_blur: bool = True,
        cancel_check: Optional[Callable[[], None]] = None,
    ):
        self.video_path = video_path
        self.seconds_step = max(0.05, float(seconds_step))
        self.max_frames = int(max(2, max_frames))
        self.extract_megapix = float(extract_megapix)

        self.similar_threshold = float(similar_threshold)
        self.similar_resize = similar_resize
        self.similar_blur = bool(similar_blur)

        self.cancel_check = cancel_check

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

    def _prep_similarity_gray(self, bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, self.similar_resize, interpolation=cv2.INTER_AREA)
        if self.similar_blur:
            g = cv2.GaussianBlur(g, (5, 5), 0)
        return g

    def too_similar(self, a_bgr: np.ndarray, b_bgr: np.ndarray) -> bool:
        """
        Cheap similarity metric: mean absolute difference on resized grayscale.
        Blur reduces sensitivity to noise/compression artifacts.
        """
        a = self._prep_similarity_gray(a_bgr)
        b = self._prep_similarity_gray(b_bgr)
        diff = cv2.mean(cv2.absdiff(a, b))[0]
        return diff < self.similar_threshold

    def extract(self, on_progress: Optional[Callable[[float, str], None]] = None) -> List[np.ndarray]:
        if not self.video_path or not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                fps = 30.0

            duration_s = (total_frames / fps) if total_frames > 0 else 0.0

            frames: List[np.ndarray] = []
            last_kept: Optional[np.ndarray] = None
            t = 0.0

            while True:
                if self.cancel_check:
                    self.cancel_check()

                if duration_s > 0 and t > duration_s:
                    break

                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame = self._downscale_to_megapix(frame, self.extract_megapix)

                if last_kept is not None and self.too_similar(last_kept, frame):
                    t += self.seconds_step
                    continue

                frames.append(frame)
                last_kept = frame

                if on_progress and duration_s > 0:
                    on_progress(min(0.95, t / duration_s), f"Extracting frames... {len(frames)}")

                if len(frames) >= self.max_frames:
                    break

                t += self.seconds_step

            if not frames:
                raise RuntimeError("No frames extracted. Try a different video or smaller seconds_step.")

            if on_progress:
                on_progress(1.0, f"Extracted {len(frames)} frames")

            return frames

        finally:
            cap.release()