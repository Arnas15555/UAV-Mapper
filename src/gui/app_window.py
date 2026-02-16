from __future__ import annotations

import os
import cv2
import numpy as np

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QMessageBox, QHBoxLayout, QCheckBox, QLineEdit
)
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, QGroupBox

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
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

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

        # ---- Params (UI) ----
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["scans", "panorama"])
        self.cmb_mode.setToolTip("OpenCV stitch mode. 'scans' often works better for planar-ish motion.")

        self.spin_seconds_step = QDoubleSpinBox()
        self.spin_seconds_step.setRange(0.05, 5.0)
        self.spin_seconds_step.setSingleStep(0.05)
        self.spin_seconds_step.setDecimals(2)
        self.spin_seconds_step.setValue(0.50)

        self.spin_max_frames = QSpinBox()
        self.spin_max_frames.setRange(2, 500)
        self.spin_max_frames.setSingleStep(5)
        self.spin_max_frames.setValue(30)

        self.spin_work_megapix = QDoubleSpinBox()
        self.spin_work_megapix.setRange(0.2, 10.0)
        self.spin_work_megapix.setSingleStep(0.1)
        self.spin_work_megapix.setDecimals(2)
        self.spin_work_megapix.setValue(1.50)

        self.spin_min_keypoints = QSpinBox()
        self.spin_min_keypoints.setRange(50, 5000)
        self.spin_min_keypoints.setSingleStep(50)
        self.spin_min_keypoints.setValue(120)

        self.spin_orb_nfeatures = QSpinBox()
        self.spin_orb_nfeatures.setRange(200, 20000)
        self.spin_orb_nfeatures.setSingleStep(200)
        self.spin_orb_nfeatures.setValue(2000)

        self.spin_extract_megapix = QDoubleSpinBox()
        self.spin_extract_megapix.setRange(0.2, 10.0)
        self.spin_extract_megapix.setSingleStep(0.1)
        self.spin_extract_megapix.setDecimals(2)
        self.spin_extract_megapix.setValue(2.0)


        # Viewer
        self.viewer = MapGraphicsView()

        # Layout
        top = QHBoxLayout()
        top.addWidget(self.btn_select)
        top.addWidget(self.btn_run)
        top.addWidget(self.btn_cancel)
        top.addSpacing(12)
        top.addWidget(self.btn_fit)
        top.addSpacing(12)
        top.addWidget(self.chk_markers)
        top.addWidget(self.marker_label)
        top.addStretch(1)

        params_form = QFormLayout()
        params_form.setContentsMargins(0, 0, 0, 0)
        params_form.addRow("Mode", self.cmb_mode)
        params_form.addRow("Seconds step", self.spin_seconds_step)
        params_form.addRow("Max frames", self.spin_max_frames)
        params_form.addRow("Work MP", self.spin_work_megapix)
        params_form.addRow("Min keypoints", self.spin_min_keypoints)
        params_form.addRow("ORB features", self.spin_orb_nfeatures)
        params_form.addRow("Extract MP", self.spin_extract_megapix)


        params_box = QGroupBox("Parameters")
        params_box.setLayout(params_form)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(params_box)
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
        self.btn_cancel.clicked.connect(self.cancel_pipeline)


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

    def cancel_pipeline(self):
        if self.worker and self.worker.isRunning():
            self.status.setText("Cancelling...")
            self.worker.requestInterruption()
            self.btn_cancel.setEnabled(False)


    def run_pipeline(self):
        if not self.video_path:
            return

        self.btn_run.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Processing...")
        for w in [self.cmb_mode, self.spin_seconds_step, self.spin_max_frames,
                  self.spin_work_megapix, self.spin_min_keypoints, self.spin_orb_nfeatures,
                  self.spin_extract_megapix]:
            w.setEnabled(False)


        seconds_step = float(self.spin_seconds_step.value())
        max_frames = int(self.spin_max_frames.value())

        mode = self.cmb_mode.currentText().strip().lower()
        work_megapix = float(self.spin_work_megapix.value())
        min_keypoints = int(self.spin_min_keypoints.value())
        orb_nfeatures = int(self.spin_orb_nfeatures.value())
        extract_megapix = float(self.spin_extract_megapix.value())

        self.worker = PipelineWorker(
            self.video_path,
            seconds_step=seconds_step,
            max_frames=max_frames,
            stitch_mode=mode,
            work_megapix=work_megapix,
            min_keypoints=min_keypoints,
            orb_nfeatures=orb_nfeatures,
            extract_megapix=extract_megapix,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished_ok)
        self.worker.finished_err.connect(self.on_finished_err)
        self.worker.start()
        self.btn_cancel.setEnabled(True)

    def on_progress(self, p: float, msg: str):
        self.progress.setValue(int(p * 100))
        self.status.setText(msg)

    def on_finished_ok(self, pano):
        base_out = os.path.join(os.getcwd(), "stitched_map.png")
        out_path = next_available_path(base_out)
        cv2.imwrite(out_path, pano)

        preview = pano
        max_preview_w = 1920
        h, w = preview.shape[:2]
        if w > max_preview_w:
            scale = max_preview_w / float(w)
            preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        pix = cv_to_qpixmap_bgr(preview)
        self.viewer.set_map_pixmap(pix)
        self.current_pano = None

        self.status.setText(f"Done. Saved: {out_path}")
        self.progress.setValue(100)

        # Enable viewer tools
        self.btn_fit.setEnabled(True)
        self.chk_markers.setEnabled(True)
        self.marker_label.setEnabled(True)

        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        for w in [self.cmb_mode, self.spin_seconds_step, self.spin_max_frames,
                  self.spin_work_megapix, self.spin_min_keypoints, self.spin_orb_nfeatures,
                  self.spin_extract_megapix]:
            w.setEnabled(True)



    def on_finished_err(self, err: str):
        QMessageBox.critical(self, "Error", err)
        self.status.setText("Failed.")
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        for w in [self.cmb_mode, self.spin_seconds_step, self.spin_max_frames,
                  self.spin_work_megapix, self.spin_min_keypoints, self.spin_orb_nfeatures,
                  self.spin_extract_megapix]:
            w.setEnabled(True)



    def on_marker_added(self, x: float, y: float):
        label = self.marker_label.text().strip()
        self.viewer.add_marker(x, y, label=label)
        if label:
            self.status.setText(f"Marker added at ({x:.0f}, {y:.0f}) label='{label}'")
        else:
            self.status.setText(f"Marker added at ({x:.0f}, {y:.0f})")