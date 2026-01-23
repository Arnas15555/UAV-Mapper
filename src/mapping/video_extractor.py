from __future__ import annotations
import os
import cv2


class VideoExtractor:
    """
    Extract frames using time-based sampling (robust when FPS metadata is weird).
    """

    def __init__(self, video_path: str, seconds_step: float = 0.15, max_frames: int = 120):
        self.video_path = video_path
        self.seconds_step = max(0.05, float(seconds_step))
        self.max_frames = int(max_frames)

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
        t = 0.0
        kept = 0

        while True:
            if duration_s > 0 and t > duration_s:
                break

            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break

            frames.append(frame)
            kept += 1

            if on_progress and duration_s > 0:
                on_progress(min(0.95, t / duration_s), f"Extracting frames... {kept}")

            if kept >= self.max_frames:
                break

            t += self.seconds_step

        cap.release()

        if not frames:
            raise RuntimeError("No frames extracted. Try a different video.")

        if on_progress:
            on_progress(1.0, f"Extracted {len(frames)} frames")

        return frames
