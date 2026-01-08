from __future__ import annotations
import os
import cv2

class VideoExtractor:
    """
    Extract frames from a video at an approx sample rate
    """
    def __init__(self, video_path: str, fps_sample: float = 2.0, max_frames: int = 200):
        self.video_path = video_path
        self.fps_sample = fps_sample
        self.max_frames = max_frames

    def extract(self, on_progress = None):
        if not self.video_path or not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            src_fps = 30.0

        step = max(1, int(round(src_fps / self.fps_sample)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        idx = 0
        kept = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if idx % step == 0:
                frames.append(frame)
                kept += 1

                if on_progress and total_frames > 0:
                    on_progress(min(0.95, idx / total_frames), f"Extracting frames... {kept}")

                if kept >= self.max_frames:
                    break
            idx += 1

        cap.release()

        if not frames:
            raise RuntimeError("No frames extracted. Try a different video or lower fps_sample.")

        if on_progress:
            on_progress(1.0, f"Extracted {len(frames)} frames")

        return frames