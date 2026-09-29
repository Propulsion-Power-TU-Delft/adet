"""Test view for the meridional profile."""

import math

from PyQt6.QtCore import QEasingCurve, QPointF, QRectF, QVariantAnimation
from PyQt6.QtGui import QBrush, QColor, QKeySequence, QPainter, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from adet.gui.entities import (
    AlignedPoints,
    DraggablePoint,
    MeridionalProfile,
    ParabolicLine,
)


ZOOM_STEP = 1.25
ZOOM_DURATION_MS = 150


class AngleLabel(QGraphicsSimpleTextItem):
    """A number label next to a point that shows an angle in degrees."""

    def __init__(self, anchor: DraggablePoint, get_angle, offset: QPointF, sources):
        super().__init__()
        self.anchor = anchor
        self.get_angle = get_angle
        self.offset = offset
        self.setBrush(QBrush(QColor('white')))
        # Keep the text size constant while the view is scaled
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        for source in sources:
            source.add_dependent(self)
        self._on_point_moved()

    def _on_point_moved(self):
        """Refresh text and position when the parabola changes."""
        self.setText(f'{self.get_angle():.1f}°')
        self.setPos(self.anchor.get_position() + self.offset)


class SyncedView(QGraphicsView):
    """A fixed-scale view sharing its x range and scroll position with its group.

    Views in the same group always have the same x range and horizontal scroll
    position, and none of them rescales, so points aligned in x stay aligned on screen.
    """

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Default extent; the scene grows beyond it when items are dragged outside
        self._scene = scene
        self._base_rect = scene.sceneRect()
        self.group: list[SyncedView] = [self]
        self._syncing_scroll = False
        self._hbar().valueChanged.connect(self._sync_scroll)

    def _hbar(self) -> QScrollBar:
        bar = self.horizontalScrollBar()
        assert bar is not None
        return bar

    def _desired_rect(self) -> QRectF:
        """Default extent grown to contain every item."""
        margin = 20
        items_rect = self._scene.itemsBoundingRect().adjusted(
            -margin, -margin, margin, margin
        )
        return self._base_rect.united(items_rect)

    def _update_extent(self):
        """Grow (or shrink back) the scene rects to contain every item,
        keeping the scale.

        The y range is per view, the x range is shared by the whole group.
        """
        rects = [view._desired_rect() for view in self.group]
        left = min(rect.left() for rect in rects)
        right = max(rect.right() for rect in rects)
        for view, rect in zip(self.group, rects):
            new_rect = QRectF(left, rect.top(), right - left, rect.height())
            if new_rect != view.sceneRect():
                view._scene.setSceneRect(new_rect)

    def fit_scale(self) -> float:
        """Scale at which the whole scene rect fits in the viewport."""
        rect = self.sceneRect()
        viewport = self.viewport()
        assert viewport is not None
        return min(viewport.width() / rect.width(), viewport.height() / rect.height())

    def _sync_scroll(self, value: int):
        if self._syncing_scroll:
            return
        for view in self.group:
            if view is not self:
                view._syncing_scroll = True
                view._hbar().setValue(value)
                view._syncing_scroll = False

    def showEvent(self, event):
        super().showEvent(event)
        # Start centered on the scene; the scale is never changed afterwards
        self._update_extent()
        self.centerOn(self.sceneRect().center())

    def mouseMoveEvent(self, event):
        """Update the scene while dragging."""
        scene = self.scene()
        if scene:
            scene.update()
        super().mouseMoveEvent(event)
        self._update_extent()
        # Scroll to follow the dragged point without changing the scale
        if scene and (item := scene.mouseGrabberItem()):
            self.ensureVisible(item, 20, 20)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._update_extent()


class MeridionalProfileTestView(QWidget):
    """Meridional profile and parabola in two separate views, so lines never overlap."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Meridional Profile Test')
        self.setGeometry(100, 100, 800, 700)

        # Meridional profile in its own scene (drawing spans x 150..550, y 50..550)
        profile_scene = QGraphicsScene()
        self.profile = MeridionalProfile(
            center1=(150, 350),
            end1=(150, 150),
            center2=(550, 250),
            end2=(550, 50),
        )
        self.profile.add_to_scene(profile_scene)
        profile_scene.setSceneRect(-100, -50, 900, 650)

        # Parabola in a second scene; the points only share x with the centers
        # through the alignment, which works across scenes
        parabola_scene = QGraphicsScene()
        center1 = self.profile.center1
        center2 = self.profile.center2
        start = DraggablePoint(center1.pos().x(), 100)
        end = DraggablePoint(center2.pos().x(), 100)
        self.start_alignment = AlignedPoints(center1, start, 'y')
        self.end_alignment = AlignedPoints(center2, end, 'y')
        control = DraggablePoint((start.pos().x() + end.pos().x()) / 2, 150)
        self.parabola = ParabolicLine(start, control, end, show_control_polygon=True)
        for item in (start, end, control, self.parabola):
            parabola_scene.addItem(item)

        # Angle counters next to the first and last point of the parabola
        sources = (start, control, end)
        self.inlet_label = AngleLabel(
            start, lambda: self.parabola.inlet_angle, QPointF(-30, 10), sources
        )
        self.outlet_label = AngleLabel(
            end, lambda: self.parabola.outlet_angle, QPointF(10, 10), sources
        )
        parabola_scene.addItem(self.inlet_label)
        parabola_scene.addItem(self.outlet_label)
        parabola_scene.setSceneRect(-100, 0, 900, 250)

        self.profile_view = SyncedView(profile_scene)
        self.parabola_view = SyncedView(parabola_scene)
        layout = QVBoxLayout(self)
        # Button row on the top right
        top_row = QHBoxLayout()
        top_row.addStretch()
        fit_button = QPushButton('Fit both views')
        fit_button.clicked.connect(self.fit_views)
        top_row.addWidget(fit_button)
        layout.addLayout(top_row)
        layout.addWidget(self.profile_view, 3)
        layout.addWidget(self.parabola_view, 1)
        # Same scale and x range, so aligned points line up on screen
        self.group = [self.profile_view, self.parabola_view]
        for view in self.group:
            view.group = self.group

        # Animation runs on the log of the zoom so steps compose smoothly
        self._zoom_applied = 0.0
        self._zoom_target = 0.0
        self._zoom_anim = QVariantAnimation(self)
        self._zoom_anim.setDuration(ZOOM_DURATION_MS)
        self._zoom_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._zoom_anim.valueChanged.connect(self._on_zoom_step)

        # Ctrl + / Ctrl - zoom both views together (Ctrl+= for keyboards without a plus key)
        for keys, factor in (
            (('Ctrl++', 'Ctrl+='), ZOOM_STEP),
            (('Ctrl+-',), 1 / ZOOM_STEP),
        ):
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(lambda f=factor: self.zoom_views(f))

    def zoom_views(self, factor: float):
        """Animate a zoom of every view by the same factor.

        Presses during a running animation add to what is left of it.
        """
        remaining = 0.0
        if self._zoom_anim.state() == QVariantAnimation.State.Running:
            remaining = self._zoom_target - self._zoom_applied
            self._zoom_anim.stop()
        self._zoom_applied = 0.0
        self._zoom_target = remaining + math.log(factor)
        self._zoom_anim.setStartValue(0.0)
        self._zoom_anim.setEndValue(self._zoom_target)
        self._zoom_anim.start()

    def _on_zoom_step(self, value):
        """Apply the part of the zoom animation done since the last step."""
        step = float(value) - self._zoom_applied
        self._zoom_applied = float(value)
        self._apply_zoom(math.exp(step))

    def _apply_zoom(self, factor: float):
        """Zoom every view by the same factor, keeping the shared x centre."""
        viewport = self.profile_view.viewport()
        assert viewport is not None
        center_x = self.profile_view.mapToScene(viewport.rect().center()).x()
        for view in self.group:
            view_viewport = view.viewport()
            assert view_viewport is not None
            center_y = view.mapToScene(view_viewport.rect().center()).y()
            view.scale(factor, factor)
            view.centerOn(center_x, center_y)

    def fit_views(self):
        """Fit the whole content of both views using one common scale."""
        self.profile_view._update_extent()
        scale = min(view.fit_scale() for view in self.group)
        for view in self.group:
            view.resetTransform()
            view.scale(scale, scale)
            view.centerOn(view.sceneRect().center())


def main():
    """Run the test application."""
    app = QApplication([])
    view = MeridionalProfileTestView()
    view.show()
    app.exec()


if __name__ == '__main__':
    main()
