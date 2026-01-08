from __future__ import annotations
import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QMessageBox, QHBoxLayout
)

from mapping.video_extractor import VideoExtractor
from mapping.stitcher import SimpleStitcher


def cv_to_qimage_bgr(mat_bgr: np.ndarray) -> QImage:
    h, w = mat_bgr.shape[:2]
    rgb = cv2.cvtColor(mat_bgr, cv2.COLOR_BGR2RGB)
    bytes_per_line = 3 * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


class PipelineWorker(QThread):
    progress = Signal(float, str)   # value, message
    finished_ok = Signal(object)    # pano (np.ndarray)
    finished_err = Signal(str)

    def __init__(self, video_path: str, fps_sample: float, max_frames: int):
        super().__init__()
        self.video_path = video_path
        self.fps_sample = fps_sample
        self.max_frames = max_frames

    def run(self):
        try:
            def cb(p, msg):
                self.progress.emit(float(p), str(msg))

            self.progress.emit(0.0, "Starting...")

            extractor = VideoExtractor(self.video_path, fps_sample=self.fps_sample, max_frames=self.max_frames)
            frames = extractor.extract(on_progress=lambda p, m: cb(p * 0.5, m))  # 0–50%

            stitcher = SimpleStitcher()
            pano = stitcher.stitch(frames, on_progress=lambda p, m: cb(0.5 + p * 0.5, m))  # 50–100%

            self.finished_ok.emit(pano)

        except Exception as e:
            self.finished_err.emit(str(e))


class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Mapper")
        self.resize(1200, 800)

        self.video_path = ""

        self.btn_select = QPushButton("Select Video")
        self.btn_run = QPushButton("Generate Map")
        self.btn_run.setEnabled(False)

        self.status = QLabel("Select a video to begin.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.image_label = QLabel("Map preview will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #111; color: #bbb; padding: 12px;")
        self.image_label.setMinimumHeight(400)

        top = QHBoxLayout()
        top.addWidget(self.btn_select)
        top.addWidget(self.btn_run)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.image_label, stretch=1)
        self.setLayout(layout)

        self.btn_select.clicked.connect(self.select_video)
        self.btn_run.clicked.connect(self.run_pipeline)

        self.worker = None

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            os.path.expanduser("~"),
            "Video Files (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI);;All Files (*)"
        )
        if not path:
            return

        self.video_path = path
        self.status.setText(f"Selected: {path}")
        self.btn_run.setEnabled(True)

    def run_pipeline(self):
        if not self.video_path:
            return

        self.btn_run.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Processing...")

        fps_sample = 1.0
        max_frames = 150

        self.worker = PipelineWorker(self.video_path, fps_sample=fps_sample, max_frames=max_frames)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished_ok)
        self.worker.finished_err.connect(self.on_finished_err)
        self.worker.start()

    def on_progress(self, p: float, msg: str):
        self.progress.setValue(int(p * 100))
        self.status.setText(msg)

    def on_finished_ok(self, pano):
        out_path = os.path.join(os.getcwd(), "stitched_map.png")
        cv2.imwrite(out_path, pano)

        qimg = cv_to_qimage_bgr(pano)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.status.setText(f"Done. Saved: {out_path}")
        self.progress.setValue(100)
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)

    def on_finished_err(self, err: str):
        QMessageBox.critical(self, "Error", err)
        self.status.setText("Failed.")
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
