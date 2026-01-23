from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

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
    - Wheel zoom toward cursor
    - Left-click drag to pan
    - Optional marker placement
    """
    marker_added = Signal(float, float)  # scene coords

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None

        # Rendering
        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Interaction state
        self._panning = False
        self._pan_start = QPoint()
        self._place_markers = False

        # Zoom config
        self._zoom = 0
        self._zoom_step = 1.25
        self._zoom_min = -10
        self._zoom_max = 30

        # Markers
        self._markers: List[MapMarker] = []

        # Behavior for panning/zooming
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)  # we will anchor manually at cursor
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.setBackgroundBrush(QBrush(Qt.GlobalColor.black))

    def set_map_pixmap(self, pixmap: QPixmap):
        self._scene.clear()
        self._markers.clear()
        self._zoom = 0

        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)

        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_view()

    def fit_to_view(self):
        if not self._pixmap_item:
            return
        self.resetTransform()
        self._zoom = 0
        self.fitInView(self._pixmap_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_place_markers(self, enabled: bool):
        self._place_markers = bool(enabled)

    def add_marker(self, x: float, y: float, label: str = ""):
        self._markers.append(MapMarker(x=x, y=y, label=label))

        r = 8.0
        pen = QPen(Qt.GlobalColor.cyan)
        pen.setWidth(3)

        dot = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        dot.setPen(pen)
        dot.setBrush(QBrush(Qt.GlobalColor.transparent))
        dot.setPos(QPointF(x, y))
        dot.setZValue(10)
        dot.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations, True)

        self._scene.addItem(dot)

        if label:
            text = QGraphicsSimpleTextItem(label)
            text.setBrush(QBrush(Qt.GlobalColor.cyan))
            text.setPos(QPointF(x + 10, y - 10))
            text.setZValue(11)
            text.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            self._scene.addItem(text)


    # ---------- Events ----------

    def wheelEvent(self, event: QWheelEvent):
        if not self._pixmap_item:
            return

        # Zoom in/out
        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom_in = delta > 0
        if zoom_in and self._zoom >= self._zoom_max:
            return
        if (not zoom_in) and self._zoom <= self._zoom_min:
            return

        # Zoom anchored at cursor position
        old_pos = self.mapToScene(event.position().toPoint())

        factor = self._zoom_step if zoom_in else (1.0 / self._zoom_step)
        self.scale(factor, factor)
        self._zoom += 1 if zoom_in else -1

        new_pos = self.mapToScene(event.position().toPoint())
        delta_scene = new_pos - old_pos
        self.translate(delta_scene.x(), delta_scene.y())

    def mousePressEvent(self, event: QMouseEvent):
        if not self._pixmap_item:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            if self._place_markers:
                self.marker_added.emit(scene_pos.x(), scene_pos.y())
                event.accept()
                return

            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)