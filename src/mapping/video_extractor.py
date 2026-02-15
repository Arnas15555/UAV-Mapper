from __future__ import annotations
import os
import cv2
import numpy as np


class VideoExtractor:
    """
    Extract frames using time-based sampling + similarity filtering.
    """

    def __init__(self, video_path: str, seconds_step: float = 0.15, max_frames: int = 120):
        self.video_path = video_path
        self.seconds_step = max(0.05, float(seconds_step))
        self.max_frames = int(max_frames)

    @staticmethod
    def too_similar(a_bgr, b_bgr, threshold=6.0):
        a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
        a = cv2.resize(a, (320, 180))
        b = cv2.resize(b, (320, 180))
        diff = cv2.mean(cv2.absdiff(a, b))[0]
        return diff < threshold

    def extract(self, on_progress=None):
        if not self.video_path or not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        duration_s = (total_frames / fps) if total_frames > 0 else 0.0

        frames = []
        last_kept = None
        t = 0.0

        while True:
            if duration_s > 0 and t > duration_s:
                break

            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break

            if last_kept is not None:
                if self.too_similar(last_kept, frame):
                    t += self.seconds_step
                    continue

            frames.append(frame)
            last_kept = frame

            if on_progress and duration_s > 0:
                on_progress(min(0.95, t / duration_s), f"Extracting frames... {len(frames)}")

            if len(frames) >= self.max_frames:
                break

            t += self.seconds_step

        cap.release()

        if not frames:
            raise RuntimeError("No frames extracted. Try a different video.")

        if on_progress:
            on_progress(1.0, f"Extracted {len(frames)} frames")

        return frames
