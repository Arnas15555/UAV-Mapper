from __future__ import annotations

import os
import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QMessageBox, QCheckBox, QLineEdit,
    QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, QGroupBox,
    QScrollArea, QFrame, QSizePolicy, QSpacerItem,
)

from gui.map_view import MapGraphicsView
from utils.pipeline_worker import PipelineWorker, SaveWorker, next_available_path
from utils.config import DEFAULTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cv_to_qpixmap_bgr(mat_bgr: np.ndarray) -> QPixmap:
    h, w = mat_bgr.shape[:2]
    rgb = cv2.cvtColor(mat_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _make_group(title: str, form: QFormLayout) -> QGroupBox:
    box = QGroupBox(title)
    box.setLayout(form)
    box.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            font-size: 11px;
            border: 1px solid #444;
            border-radius: 6px;
            margin-top: 8px;
            padding: 4px;
            color: #ccc;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
    """)
    return box


def _form() -> QFormLayout:
    f = QFormLayout()
    f.setContentsMargins(6, 6, 6, 6)
    f.setSpacing(6)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    return f


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

_BG      = "#1e1e2e"
_SIDEBAR = "#181825"
_ACCENT  = "#89b4fa"
_GREEN   = "#a6e3a1"
_RED     = "#f38ba8"
_TEXT    = "#cdd6f4"
_SUBTEXT = "#a6adc8"
_BORDER  = "#313244"


def _btn(color: str, text_color: str = "#1e1e2e") -> str:
    return f"""
        QPushButton {{
            background-color: {color};
            color: {text_color};
            border: none;
            border-radius: 5px;
            padding: 6px 14px;
            font-weight: bold;
            font-size: 12px;
        }}
        QPushButton:hover {{ background-color: {color}cc; }}
        QPushButton:pressed {{ background-color: {color}88; }}
        QPushButton:disabled {{ background-color: #313244; color: #585b70; }}
    """


class AppWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Mapper")
        self.resize(1400, 860)
        self.setMinimumSize(900, 600)
        self.video_path = ""

        self._build_widgets()
        self._apply_styles()
        self._build_layout()
        self._connect_signals()

        self.worker: PipelineWorker | None = None
        self._save_worker: SaveWorker | None = None
        self._last_pano: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _build_widgets(self):
        # Toolbar buttons
        self.btn_select = QPushButton("📂  Select Video")
        self.btn_run    = QPushButton("▶  Generate Map")
        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_fit    = QPushButton("⊡  Fit to View")
        self.btn_save   = QPushButton("💾  Save As…")

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_fit.setEnabled(False)
        self.btn_save.setEnabled(False)

        # Marker controls
        self.chk_markers = QCheckBox("Place markers")
        self.chk_markers.setEnabled(False)
        self.marker_label = QLineEdit()
        self.marker_label.setPlaceholderText("Marker label…")
        self.marker_label.setEnabled(False)
        self.marker_label.setFixedWidth(130)

        # Status + progress
        self.status = QLabel("Select a video to begin.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        self.progress.setFixedWidth(150)

        # ── Extraction params ────────────────────────────────────────
        self.spin_seconds_step = QDoubleSpinBox()
        self.spin_seconds_step.setRange(0.05, 5.0)
        self.spin_seconds_step.setSingleStep(0.05)
        self.spin_seconds_step.setDecimals(2)
        self.spin_seconds_step.setValue(DEFAULTS["seconds_step"])
        self.spin_seconds_step.setToolTip("Seconds between sampled frames. Lower = denser sampling.")

        self.spin_max_frames = QSpinBox()
        self.spin_max_frames.setRange(2, 500)
        self.spin_max_frames.setSingleStep(5)
        self.spin_max_frames.setValue(DEFAULTS["max_frames"])
        self.spin_max_frames.setToolTip("Hard cap on frames extracted from the video.")

        self.spin_extract_megapix = QDoubleSpinBox()
        self.spin_extract_megapix.setRange(0.2, 10.0)
        self.spin_extract_megapix.setSingleStep(0.1)
        self.spin_extract_megapix.setDecimals(2)
        self.spin_extract_megapix.setValue(DEFAULTS["extract_megapix"])
        self.spin_extract_megapix.setToolTip("Downscale frames to this MP at extraction time.")

        self.spin_similar_threshold = QDoubleSpinBox()
        self.spin_similar_threshold.setRange(1.0, 50.0)
        self.spin_similar_threshold.setSingleStep(0.5)
        self.spin_similar_threshold.setDecimals(1)
        self.spin_similar_threshold.setValue(DEFAULTS["similar_threshold"])
        self.spin_similar_threshold.setToolTip(
            "Frames with pixel diff below this are dropped as duplicates.\n"
            "Raise to keep more frames, lower to filter more aggressively."
        )

        # ── Stitching params ─────────────────────────────────────────
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["panorama", "scans"])
        self.cmb_mode.setCurrentText(DEFAULTS["stitch_mode"])
        self.cmb_mode.setToolTip("panorama = spherical warp (best for UAV). scans = flat document mode.")

        self.spin_work_megapix = QDoubleSpinBox()
        self.spin_work_megapix.setRange(0.2, 10.0)
        self.spin_work_megapix.setSingleStep(0.1)
        self.spin_work_megapix.setDecimals(2)
        self.spin_work_megapix.setValue(DEFAULTS["work_megapix"])
        self.spin_work_megapix.setToolTip("Internal stitching resolution. Higher = sharper but slower.")

        self.spin_orb_nfeatures = QSpinBox()
        self.spin_orb_nfeatures.setRange(200, 20000)
        self.spin_orb_nfeatures.setSingleStep(200)
        self.spin_orb_nfeatures.setValue(DEFAULTS["orb_nfeatures"])
        self.spin_orb_nfeatures.setToolTip("ORB keypoints per frame. More = robust but slower.")

        self.spin_min_keypoints = QSpinBox()
        self.spin_min_keypoints.setRange(10, 5000)
        self.spin_min_keypoints.setSingleStep(10)
        self.spin_min_keypoints.setValue(DEFAULTS["min_keypoints"])
        self.spin_min_keypoints.setToolTip(
            "Frames with fewer than this many keypoints are discarded.\n"
            "If too many frames are rejected, lower this value."
        )

        # ── Frame selection params ───────────────────────────────────
        self.spin_min_motion = QDoubleSpinBox()
        self.spin_min_motion.setRange(0.0, 50.0)
        self.spin_min_motion.setSingleStep(0.5)
        self.spin_min_motion.setDecimals(1)
        self.spin_min_motion.setValue(DEFAULTS["min_motion_px"])
        self.spin_min_motion.setToolTip(
            "Frame pairs with less motion than this (px) are rejected.\n"
            "For slow/hovering UAV, set to 2–3."
        )

        self.spin_target_motion = QDoubleSpinBox()
        self.spin_target_motion.setRange(1.0, 200.0)
        self.spin_target_motion.setSingleStep(1.0)
        self.spin_target_motion.setDecimals(1)
        self.spin_target_motion.setValue(DEFAULTS["target_motion_px"])
        self.spin_target_motion.setToolTip(
            "Ideal inter-frame motion in pixels. Frames near this score higher.\n"
            "Good starting point: ~10–15% of your frame width."
        )

        self.spin_max_stitch_frames = QSpinBox()
        self.spin_max_stitch_frames.setRange(2, 500)
        self.spin_max_stitch_frames.setSingleStep(10)
        self.spin_max_stitch_frames.setValue(DEFAULTS["max_frames_for_stitch"])
        self.spin_max_stitch_frames.setToolTip(
            "How many frames are passed to the stitcher after selection.\n"
            "More = better coverage but more RAM + time."
        )

        # Map viewer
        self.viewer = MapGraphicsView()

        # Reset button
        self.btn_reset = QPushButton("↺  Reset Defaults")

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {_BG};
                color: {_TEXT};
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 12px;
            }}
            QLabel {{ color: {_TEXT}; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: #313244;
                border: 1px solid {_BORDER};
                border-radius: 4px;
                padding: 3px 6px;
                color: {_TEXT};
                min-height: 22px;
            }}
            QLineEdit:focus, QSpinBox:focus,
            QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {_ACCENT};
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: #313244;
                selection-background-color: {_ACCENT};
                color: {_TEXT};
            }}
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 16px;
                background-color: #45475a;
                border: none;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {_ACCENT};
            }}
            QProgressBar {{
                background-color: #313244;
                border: 1px solid {_BORDER};
                border-radius: 4px;
                text-align: center;
                color: transparent;
            }}
            QProgressBar::chunk {{
                background-color: {_ACCENT};
                border-radius: 4px;
            }}
            QScrollBar:vertical {{
                background: {_SIDEBAR};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #45475a;
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
            QCheckBox {{ color: {_TEXT}; spacing: 6px; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border-radius: 3px; border: 1px solid #555;
                background: #313244;
            }}
            QCheckBox::indicator:checked {{
                background: {_ACCENT}; border: 1px solid {_ACCENT};
            }}
        """)
        self.btn_select.setStyleSheet(_btn(_ACCENT))
        self.btn_run.setStyleSheet(_btn(_GREEN))
        self.btn_cancel.setStyleSheet(_btn(_RED))
        self.btn_fit.setStyleSheet(_btn("#cba6f7"))
        self.btn_save.setStyleSheet(_btn("#fab387"))
        self.btn_reset.setStyleSheet(_btn("#45475a", _TEXT))
        self.status.setStyleSheet(f"color: {_SUBTEXT}; font-size: 11px;")
        self.viewer.setStyleSheet(f"background: #11111b; border: 1px solid {_BORDER}; border-radius: 6px;")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        # ── Toolbar ──────────────────────────────────────────────────
        logo = QLabel("✈  UAV Mapper")
        logo.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {_ACCENT}; letter-spacing: 1px;")

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {_BORDER};")

        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(8)
        toolbar_row.setContentsMargins(10, 0, 10, 0)
        toolbar_row.addWidget(logo)
        toolbar_row.addWidget(sep)
        toolbar_row.addWidget(self.btn_select)
        toolbar_row.addWidget(self.btn_run)
        toolbar_row.addWidget(self.btn_cancel)
        toolbar_row.addSpacing(6)
        toolbar_row.addWidget(self.btn_fit)
        toolbar_row.addWidget(self.btn_save)
        toolbar_row.addSpacing(6)
        toolbar_row.addWidget(self.chk_markers)
        toolbar_row.addWidget(self.marker_label)
        toolbar_row.addStretch(1)
        toolbar_row.addWidget(self.status)
        toolbar_row.addWidget(self.progress)

        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar_row)
        toolbar_widget.setFixedHeight(52)
        toolbar_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {_SIDEBAR};
                border: 1px solid {_BORDER};
                border-radius: 6px;
            }}
        """)

        # ── Sidebar ───────────────────────────────────────────────────
        f_extract = _form()
        f_extract.addRow("Seconds step",   self.spin_seconds_step)
        f_extract.addRow("Max frames",     self.spin_max_frames)
        f_extract.addRow("Extract MP",     self.spin_extract_megapix)
        f_extract.addRow("Similarity thr", self.spin_similar_threshold)

        f_stitch = _form()
        f_stitch.addRow("Mode",          self.cmb_mode)
        f_stitch.addRow("Work MP",       self.spin_work_megapix)
        f_stitch.addRow("ORB features",  self.spin_orb_nfeatures)
        f_stitch.addRow("Min keypoints", self.spin_min_keypoints)

        f_sel = _form()
        f_sel.addRow("Min motion px",     self.spin_min_motion)
        f_sel.addRow("Target motion px",  self.spin_target_motion)
        f_sel.addRow("Max stitch frames", self.spin_max_stitch_frames)

        inner = QWidget()
        inner.setStyleSheet(f"background: {_SIDEBAR};")
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)
        vbox.addWidget(_make_group("Frame Extraction", f_extract))
        vbox.addWidget(_make_group("Stitching", f_stitch))
        vbox.addWidget(_make_group("Frame Selection", f_sel))
        vbox.addWidget(self.btn_reset)
        vbox.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidget(inner)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(250)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {_BORDER};
                border-radius: 6px;
                background: {_SIDEBAR};
            }}
        """)

        # ── Body ─────────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(8)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(sidebar_scroll)
        body.addWidget(self.viewer, stretch=1)

        # ── Root ─────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addWidget(toolbar_widget)
        root.addLayout(body, stretch=1)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.btn_select.clicked.connect(self.select_video)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_cancel.clicked.connect(self.cancel_pipeline)
        self.btn_fit.clicked.connect(self.viewer.fit_to_view)
        self.btn_save.clicked.connect(self.save_as)
        self.btn_reset.clicked.connect(self._reset_defaults)
        self.chk_markers.toggled.connect(self.viewer.set_place_markers)
        self.viewer.marker_added.connect(self.on_marker_added)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_param_widgets(self):
        return [
            self.cmb_mode, self.spin_seconds_step, self.spin_max_frames,
            self.spin_extract_megapix, self.spin_similar_threshold,
            self.spin_work_megapix, self.spin_min_keypoints, self.spin_orb_nfeatures,
            self.spin_min_motion, self.spin_target_motion, self.spin_max_stitch_frames,
        ]

    def _set_params_enabled(self, enabled: bool):
        for w in self._all_param_widgets():
            w.setEnabled(bool(enabled))

    def _set_post_run_controls(self, enabled: bool):
        self.btn_fit.setEnabled(enabled)
        self.btn_save.setEnabled(enabled and self._last_pano is not None)
        self.chk_markers.setEnabled(enabled)
        self.marker_label.setEnabled(enabled)

    def _reset_defaults(self):
        self.cmb_mode.setCurrentText(DEFAULTS["stitch_mode"])
        self.spin_seconds_step.setValue(DEFAULTS["seconds_step"])
        self.spin_max_frames.setValue(DEFAULTS["max_frames"])
        self.spin_extract_megapix.setValue(DEFAULTS["extract_megapix"])
        self.spin_similar_threshold.setValue(DEFAULTS["similar_threshold"])
        self.spin_work_megapix.setValue(DEFAULTS["work_megapix"])
        self.spin_min_keypoints.setValue(DEFAULTS["min_keypoints"])
        self.spin_orb_nfeatures.setValue(DEFAULTS["orb_nfeatures"])
        self.spin_min_motion.setValue(DEFAULTS["min_motion_px"])
        self.spin_target_motion.setValue(DEFAULTS["target_motion_px"])
        self.spin_max_stitch_frames.setValue(DEFAULTS["max_frames_for_stitch"])

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", os.path.expanduser("~"),
            "Video Files (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI);;All Files (*)",
        )
        if not path:
            return
        self.video_path = path
        self.status.setText(f"📹 {os.path.basename(path)}")
        self.btn_run.setEnabled(True)

    def cancel_pipeline(self):
        if self.worker and self.worker.isRunning():
            self.status.setText("Cancelling…")
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
        self.status.setText("Processing…")

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
            min_motion_px=float(self.spin_min_motion.value()),
            target_motion_px=float(self.spin_target_motion.value()),
            max_frames_for_stitch=int(self.spin_max_stitch_frames.value()),
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
        preview = pano
        h, w = preview.shape[:2]
        if w > 1920:
            scale = 1920 / float(w)
            preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self.viewer.set_map_pixmap(cv_to_qpixmap_bgr(preview))
        self.progress.setValue(100)
        self.status.setText("Saving map…")

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
        self.status.setText(f"Saved: {os.path.basename(path)}")

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
        if self._last_pano is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Map As",
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
            self.status.setText(f"📍 '{label}' at ({x:.0f}, {y:.0f})")
        else:
            self.status.setText(f"📍 Marker at ({x:.0f}, {y:.0f})")
        self.marker_label.clear()