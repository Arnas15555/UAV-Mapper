from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QWheelEvent, QMouseEvent, QBrush, QPainter
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
)


class MapGraphicsView(QGraphicsView):
    """
    Zoom/pan capable map viewer based on QGraphicsView.

    - Wheel zoom toward cursor (stable anchor)
    - Pan with Left-drag or Middle-drag
    - Double-click to fit the map to the viewport
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None

        # Smooth pixmap rendering helps at non-integer zoom levels
        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.SmoothPixmapTransform
        )

        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)

        # Interaction state
        self._panning = False
        self._pan_start = QPoint()

        # Zoom config
        self._zoom_step = 1.25
        self._min_scale = 0.05
        self._max_scale = 40.0
        self._scale_factor = 1.0

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(Qt.GlobalColor.black))

        self.setMouseTracking(True)

    # ---------- Public API ----------

    def has_map(self) -> bool:
        return self._pixmap_item is not None

    def set_map_pixmap(self, pixmap: QPixmap):
        self._scene.clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_view()

    def fit_to_view(self):
        if not self._pixmap_item:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # Re-derive _scale_factor from the resulting matrix since fitInView bypasses _zoom_at.
        t = self.transform()
        self._scale_factor = float(math.sqrt(t.m11() * t.m22()))

    # ---------- Internals ----------

    def _start_pan(self, event: QMouseEvent):
        self._panning = True
        self._pan_start = event.pos()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def _end_pan(self, event: QMouseEvent):
        self._panning = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        event.accept()

    def _pan_move(self, event: QMouseEvent):
        delta = event.pos() - self._pan_start
        self._pan_start = event.pos()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        event.accept()

    def _zoom_at(self, view_pos: QPoint, factor: float):
        if not self._pixmap_item:
            return

        new_scale = self._scale_factor * factor
        if new_scale < self._min_scale:
            factor = self._min_scale / self._scale_factor
            new_scale = self._min_scale
        elif new_scale > self._max_scale:
            factor = self._max_scale / self._scale_factor
            new_scale = self._max_scale

        if abs(factor - 1.0) < 1e-6:
            return

        # Keep the scene point under the cursor fixed in the viewport
        old_scene = self.mapToScene(view_pos)
        self.scale(factor, factor)
        self._scale_factor = new_scale
        new_scene = self.mapToScene(view_pos)
        delta = new_scene - old_scene

        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + int(delta.x()))
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + int(delta.y()))

    # ---------- Events ----------

    def wheelEvent(self, event: QWheelEvent):
        if not self._pixmap_item:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self._zoom_step if delta > 0 else (1.0 / self._zoom_step)
        self._zoom_at(event.position().toPoint(), factor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap_item:
            self.fit_to_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if not self._pixmap_item:
            super().mousePressEvent(event)
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._start_pan(event)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            self._pan_move(event)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton) and self._panning:
            self._end_pan(event)
            return
        super().mouseReleaseEvent(event)
