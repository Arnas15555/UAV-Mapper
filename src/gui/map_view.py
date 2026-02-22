from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

from PySide6.QtCore import Qt, QPoint, QPointF, Signal
from PySide6.QtGui import QPixmap, QWheelEvent, QMouseEvent, QBrush, QPen, QPainter
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem,
    QGraphicsSimpleTextItem
)


@dataclass
class MapMarker:
    x: float
    y: float
    label: str = ""


class MapGraphicsView(QGraphicsView):
    """
    Zoom/pan capable map viewer based on QGraphicsView.

    - Wheel zoom toward cursor (stable anchor)
    - Pan with Left-drag (when not placing markers) or Middle-drag
    - Optional marker placement on left-click
    - Double-click to fit the map to the viewport
    - Right-click to remove nearest marker within a pixel radius
    """

    marker_added   = Signal(float, float)  # scene coords (x, y)
    marker_removed = Signal(float, float)  # scene coords (x, y) of removed marker

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
        self._place_markers = False

        # Zoom config
        self._zoom_step = 1.25
        self._min_scale = 0.05
        self._max_scale = 40.0
        self._scale_factor = 1.0  # tracks current uniform scale for clamping

        # Markers
        self._markers: List[MapMarker] = []
        self._marker_items: List[Tuple[QGraphicsEllipseItem, Optional[QGraphicsSimpleTextItem]]] = []
        self._remove_radius_px = 18.0

        # Anchoring / drag config
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
        self._markers.clear()
        self._marker_items.clear()

        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)

        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_view()

    def clear_markers(self):
        for dot, text in self._marker_items:
            self._scene.removeItem(dot)
            if text is not None:
                self._scene.removeItem(text)
        self._marker_items.clear()
        self._markers.clear()

    def fit_to_view(self):
        if not self._pixmap_item:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # fitInView modifies the transform internally without going through
        # _zoom_at, so we re-derive _scale_factor from the resulting matrix.
        # Using the geometric mean of m11/m22 handles any residual non-uniform
        # scaling more robustly than reading m11 alone.
        t = self.transform()
        self._scale_factor = float(math.sqrt(t.m11() * t.m22()))

    def set_place_markers(self, enabled: bool):
        self._place_markers = bool(enabled)

    def add_marker(self, x: float, y: float, label: str = ""):
        if not self._pixmap_item:
            return

        self._markers.append(MapMarker(x=x, y=y, label=label))

        r = 8.0
        pen = QPen(Qt.GlobalColor.cyan)
        pen.setWidth(3)

        dot = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        dot.setPen(pen)
        dot.setBrush(QBrush(Qt.GlobalColor.transparent))
        dot.setPos(QPointF(x, y))
        dot.setZValue(10)
        # Keep marker size constant in screen-space regardless of zoom level
        dot.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._scene.addItem(dot)

        text_item = None
        if label:
            text_item = QGraphicsSimpleTextItem(label)
            text_item.setBrush(QBrush(Qt.GlobalColor.cyan))
            text_item.setPos(QPointF(x + 10, y - 10))
            text_item.setZValue(11)
            text_item.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            self._scene.addItem(text_item)

        self._marker_items.append((dot, text_item))

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

    def _try_remove_nearest_marker(self, scene_pos: QPointF) -> bool:
        """Removes nearest marker if it is within _remove_radius_px screen pixels."""
        if not self._marker_items:
            return False

        best_i = -1
        best_d2 = 1e18
        sx, sy = scene_pos.x(), scene_pos.y()

        for i, (dot, _) in enumerate(self._marker_items):
            p = dot.pos()
            dx, dy = p.x() - sx, p.y() - sy
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        if best_i < 0:
            return False

        dot, text = self._marker_items[best_i]
        dot_view   = self.mapFromScene(dot.pos())
        click_view = self.mapFromScene(scene_pos)
        dv = dot_view - click_view
        dist = math.sqrt(dv.x() ** 2 + dv.y() ** 2)

        if dist > self._remove_radius_px:
            return False

        self._scene.removeItem(dot)
        if text is not None:
            self._scene.removeItem(text)

        removed = self._markers.pop(best_i)
        self._marker_items.pop(best_i)
        self.marker_removed.emit(removed.x, removed.y)
        return True

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

        if event.button() == Qt.MouseButton.LeftButton and self._place_markers:
            scene_pos = self.mapToScene(event.pos())
            self.marker_added.emit(scene_pos.x(), scene_pos.y())
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            scene_pos = self.mapToScene(event.pos())
            if self._try_remove_nearest_marker(scene_pos):
                event.accept()
                return

        if event.button() == Qt.MouseButton.MiddleButton:
            self._start_pan(event)
            return

        if event.button() == Qt.MouseButton.LeftButton and not self._place_markers:
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