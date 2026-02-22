from __future__ import annotations

import os
import cv2
import numpy as np

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QMessageBox, QHBoxLayout, QCheckBox, QLineEdit,
    QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, QGroupBox,
)

from gui.map_view import MapGraphicsView
from utils.pipeline_worker import PipelineWorker, SaveWorker, next_available_path
from utils.config import DEFAULTS


def cv_to_qpixmap_bgr(mat_bgr: np.ndarray) -> QPixmap:
    """
    Convert a BGR numpy array to a QPixmap.

    .copy() is called on the QImage to detach it from the numpy buffer —
    without this the buffer can be freed before Qt finishes rendering,
    causing crashes or rendering artifacts.
    """
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

        # ---- Action buttons ----
        self.btn_select = QPushButton("Select Video")
        self.btn_run    = QPushButton("Generate Map")
        self.btn_run.setEnabled(False)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_fit    = QPushButton("Fit to View")
        self.btn_fit.setEnabled(False)
        self.btn_save   = QPushButton("Save As…")
        self.btn_save.setEnabled(False)

        # ---- Marker controls ----
        self.chk_markers = QCheckBox("Place markers")
        self.chk_markers.setEnabled(False)
        self.marker_label = QLineEdit()
        self.marker_label.setPlaceholderText("Marker label (optional)")
        self.marker_label.setEnabled(False)

        # ---- Status / progress ----
        self.status   = QLabel("Select a video to begin.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        # ---- Parameters (initialised from shared DEFAULTS) ----
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["scans", "panorama"])
        self.cmb_mode.setCurrentText(DEFAULTS["stitch_mode"])

        self.spin_seconds_step = QDoubleSpinBox()
        self.spin_seconds_step.setRange(0.05, 5.0)
        self.spin_seconds_step.setSingleStep(0.05)
        self.spin_seconds_step.setDecimals(2)
        self.spin_seconds_step.setValue(DEFAULTS["seconds_step"])

        self.spin_max_frames = QSpinBox()
        self.spin_max_frames.setRange(2, 500)
        self.spin_max_frames.setSingleStep(5)
        self.spin_max_frames.setValue(DEFAULTS["max_frames"])

        self.spin_work_megapix = QDoubleSpinBox()
        self.spin_work_megapix.setRange(0.2, 10.0)
        self.spin_work_megapix.setSingleStep(0.1)
        self.spin_work_megapix.setDecimals(2)
        self.spin_work_megapix.setValue(DEFAULTS["work_megapix"])

        self.spin_min_keypoints = QSpinBox()
        self.spin_min_keypoints.setRange(50, 5000)
        self.spin_min_keypoints.setSingleStep(50)
        self.spin_min_keypoints.setValue(DEFAULTS["min_keypoints"])

        self.spin_orb_nfeatures = QSpinBox()
        self.spin_orb_nfeatures.setRange(200, 20000)
        self.spin_orb_nfeatures.setSingleStep(200)
        self.spin_orb_nfeatures.setValue(DEFAULTS["orb_nfeatures"])

        self.spin_extract_megapix = QDoubleSpinBox()
        self.spin_extract_megapix.setRange(0.2, 10.0)
        self.spin_extract_megapix.setSingleStep(0.1)
        self.spin_extract_megapix.setDecimals(2)
        self.spin_extract_megapix.setValue(DEFAULTS["extract_megapix"])

        self.spin_similar_threshold = QDoubleSpinBox()
        self.spin_similar_threshold.setRange(1.0, 50.0)
        self.spin_similar_threshold.setSingleStep(0.5)
        self.spin_similar_threshold.setDecimals(1)
        self.spin_similar_threshold.setValue(DEFAULTS["similar_threshold"])

        # ---- Map viewer ----
        self.viewer = MapGraphicsView()

        # ---- Layout ----
        top = QHBoxLayout()
        top.addWidget(self.btn_select)
        top.addWidget(self.btn_run)
        top.addWidget(self.btn_cancel)
        top.addSpacing(12)
        top.addWidget(self.btn_fit)
        top.addWidget(self.btn_save)
        top.addSpacing(12)
        top.addWidget(self.chk_markers)
        top.addWidget(self.marker_label)
        top.addStretch(1)

        params_form = QFormLayout()
        params_form.setContentsMargins(0, 0, 0, 0)
        params_form.addRow("Mode",           self.cmb_mode)
        params_form.addRow("Seconds step",   self.spin_seconds_step)
        params_form.addRow("Max frames",     self.spin_max_frames)
        params_form.addRow("Extract MP",     self.spin_extract_megapix)
        params_form.addRow("Similarity thr", self.spin_similar_threshold)
        params_form.addRow("Work MP",        self.spin_work_megapix)
        params_form.addRow("Min keypoints",  self.spin_min_keypoints)
        params_form.addRow("ORB features",   self.spin_orb_nfeatures)

        params_box = QGroupBox("Parameters")
        params_box.setLayout(params_form)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(params_box)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.viewer, stretch=1)
        self.setLayout(layout)

        # ---- Signals ----
        self.btn_select.clicked.connect(self.select_video)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_cancel.clicked.connect(self.cancel_pipeline)
        self.btn_fit.clicked.connect(self.viewer.fit_to_view)
        self.btn_save.clicked.connect(self.save_as)
        self.chk_markers.toggled.connect(self.viewer.set_place_markers)
        self.viewer.marker_added.connect(self.on_marker_added)

        # Internal state
        self.worker: PipelineWorker | None = None
        self._save_worker: SaveWorker | None = None
        self._last_pano: np.ndarray | None = None  # kept for "Save As"

    # ---------- helpers ----------

    def _set_params_enabled(self, enabled: bool):
        for w in [
            self.cmb_mode, self.spin_seconds_step, self.spin_max_frames,
            self.spin_extract_megapix, self.spin_similar_threshold,
            self.spin_work_megapix, self.spin_min_keypoints, self.spin_orb_nfeatures,
        ]:
            w.setEnabled(bool(enabled))

    def _set_post_run_controls(self, enabled: bool):
        self.btn_fit.setEnabled(enabled)
        self.btn_save.setEnabled(enabled and self._last_pano is not None)
        self.chk_markers.setEnabled(enabled)
        self.marker_label.setEnabled(enabled)

    # ---------- slots ----------

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            os.path.expanduser("~"),
            "Video Files (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI);;All Files (*)",
        )
        if not path:
            return
        self.video_path = path
        self.status.setText(f"Selected: {path}")
        self.btn_run.setEnabled(True)

    def cancel_pipeline(self):
        if self.worker and self.worker.isRunning():
            self.status.setText("Cancelling map generation...")
            self.worker.requestInterruption()
            self.btn_cancel.setEnabled(False)

    def run_pipeline(self):
        if not self.video_path:
            return

        self.btn_run.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._set_params_enabled(False)
        self._set_post_run_controls(False)

        self.progress.setValue(0)
        self.status.setText("Processing...")

        self.worker = PipelineWorker(
            self.video_path,
            seconds_step=float(self.spin_seconds_step.value()),
            max_frames=int(self.spin_max_frames.value()),
            stitch_mode=self.cmb_mode.currentText().strip().lower(),
            work_megapix=float(self.spin_work_megapix.value()),
            min_keypoints=int(self.spin_min_keypoints.value()),
            orb_nfeatures=int(self.spin_orb_nfeatures.value()),
            extract_megapix=float(self.spin_extract_megapix.value()),
            similar_threshold=float(self.spin_similar_threshold.value()),
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished_ok)
        self.worker.finished_err.connect(self.on_finished_err)
        self.worker.start()

    def on_progress(self, p: float, msg: str):
        self.progress.setValue(int(max(0.0, min(1.0, p)) * 100))
        self.status.setText(msg)

    def on_finished_ok(self, pano: np.ndarray):
        self._last_pano = pano

        # Build preview (downscale if very wide to save RAM / texture memory)
        preview = pano
        max_preview_w = 1920
        h, w = preview.shape[:2]
        if w > max_preview_w:
            scale = max_preview_w / float(w)
            preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        pix = cv_to_qpixmap_bgr(preview)
        self.viewer.set_map_pixmap(pix)

        self.progress.setValue(100)
        self.status.setText("Saving map…")

        # Save to disk on a background thread — avoids freezing the UI for
        # large panoramas where imwrite can take several seconds.
        base_out = os.path.join(os.getcwd(), "stitched_map.png")
        self._save_worker = SaveWorker(pano, base_out)
        self._save_worker.save_ok.connect(self._on_save_ok)
        self._save_worker.save_err.connect(self._on_save_err)
        self._save_worker.start()

        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._set_params_enabled(True)
        self._set_post_run_controls(True)

    def _on_save_ok(self, path: str):
        self.status.setText(f"Done. Saved: {path}")

    def _on_save_err(self, err: str):
        QMessageBox.warning(self, "Save Error", f"Could not save map:\n{err}")
        self.status.setText("Map generated but save failed.")

    def on_finished_err(self, err: str):
        if "Cancelled" in err:
            self.status.setText("Generation cancelled.")
        else:
            QMessageBox.critical(self, "Error", err)
            self.status.setText("Failed.")

        self.btn_run.setEnabled(bool(self.video_path))
        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._set_params_enabled(True)

    def save_as(self):
        """Explicit Save As dialog so users can choose filename/location."""
        if self._last_pano is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Map As",
            os.path.join(os.path.expanduser("~"), "stitched_map.png"),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if not path:
            return

        self.status.setText("Saving…")
        self._save_worker = SaveWorker(self._last_pano, path)
        self._save_worker.save_ok.connect(self._on_save_ok)
        self._save_worker.save_err.connect(self._on_save_err)
        self._save_worker.start()

    def on_marker_added(self, x: float, y: float):
        label = self.marker_label.text().strip()
        self.viewer.add_marker(x, y, label=label)

        if label:
            self.status.setText(f"Marker added at ({x:.0f}, {y:.0f}) label='{label}'")
        else:
            self.status.setText(f"Marker added at ({x:.0f}, {y:.0f})")

        # Clear the label field after placement so repeated clicks don't
        # silently reuse the same label
        self.marker_label.clear()