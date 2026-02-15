from __future__ import annotations

import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QMessageBox, QHBoxLayout, QCheckBox, QLineEdit
)

from gui.map_view import MapGraphicsView
from utils.pipeline_worker import PipelineWorker
from utils.pipeline_worker import next_available_path

def cv_to_qpixmap_bgr(mat_bgr: np.ndarray) -> QPixmap:
    h, w = mat_bgr.shape[:2]
    rgb = cv2.cvtColor(mat_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Mapper")
        self.resize(1280, 820)

        self.video_path = ""
        self.current_pano = None

        # Controls
        self.btn_select = QPushButton("Select Video")
        self.btn_run = QPushButton("Generate Map")
        self.btn_run.setEnabled(False)

        self.btn_fit = QPushButton("Fit to View")
        self.btn_fit.setEnabled(False)

        self.chk_markers = QCheckBox("Place markers")
        self.chk_markers.setEnabled(False)

        self.marker_label = QLineEdit()
        self.marker_label.setPlaceholderText("Marker label (optional)")
        self.marker_label.setEnabled(False)

        self.status = QLabel("Select a video to begin.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        # Viewer
        self.viewer = MapGraphicsView()

        # Layout
        top = QHBoxLayout()
        top.addWidget(self.btn_select)
        top.addWidget(self.btn_run)
        top.addSpacing(12)
        top.addWidget(self.btn_fit)
        top.addSpacing(12)
        top.addWidget(self.chk_markers)
        top.addWidget(self.marker_label)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.viewer, stretch=1)
        self.setLayout(layout)

        # Signals
        self.btn_select.clicked.connect(self.select_video)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_fit.clicked.connect(self.viewer.fit_to_view)
        self.chk_markers.toggled.connect(self.viewer.set_place_markers)
        self.viewer.marker_added.connect(self.on_marker_added)

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

        seconds_step = 0.5
        max_frames = 30

        self.worker = PipelineWorker(self.video_path, seconds_step=seconds_step, max_frames=max_frames)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished_ok)
        self.worker.finished_err.connect(self.on_finished_err)
        self.worker.start()

    def on_progress(self, p: float, msg: str):
        self.progress.setValue(int(p * 100))
        self.status.setText(msg)

    def on_finished_ok(self, pano):
        self.current_pano = pano

        base_out = os.path.join(os.getcwd(), "stitched_map.png")
        out_path = next_available_path(base_out)
        cv2.imwrite(out_path, pano)

        pix = cv_to_qpixmap_bgr(pano)
        self.viewer.set_map_pixmap(pix)

        self.status.setText(f"Done. Saved: {out_path}")
        self.progress.setValue(100)

        # Enable viewer tools
        self.btn_fit.setEnabled(True)
        self.chk_markers.setEnabled(True)
        self.marker_label.setEnabled(True)

        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)

    def on_finished_err(self, err: str):
        QMessageBox.critical(self, "Error", err)
        self.status.setText("Failed.")
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)

    def on_marker_added(self, x: float, y: float):
        label = self.marker_label.text().strip()
        self.viewer.add_marker(x, y, label=label)
        if label:
            self.status.setText(f"Marker added at ({x:.0f}, {y:.0f}) label='{label}'")
        else:
            self.status.setText(f"Marker added at ({x:.0f}, {y:.0f})")