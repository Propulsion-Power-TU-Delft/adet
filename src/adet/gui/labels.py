"""Annotation items of the meridional view: angle labels, rotation arrow, radius axis."""

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsSimpleTextItem,
)

from adet.gui.entities import DraggablePoint, ParabolicLine


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


class RotationArrow(QGraphicsItem):
    """A vertical arrow showing the direction of rotation.

    The vertical direction of the view is the tangential one, so the arrow points up
    for positive ``omega`` and down for negative. Its size stays constant while the
    view is scaled.
    """

    LENGTH = 70  # pixels
    HEAD = 10  # pixels
    COLOR = QColor(255, 200, 50)

    def __init__(self, omega: float, parabola: ParabolicLine, y: float):
        super().__init__()
        self.omega = omega
        self.parabola = parabola
        self.fixed_y = y
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.label = QGraphicsSimpleTextItem('rotation', self)
        self.label.setBrush(QBrush(self.COLOR))
        self.label.setPos(-self.label.boundingRect().width() / 2, self.LENGTH / 2 + 4)
        parabola.start_point.add_dependent(self)
        parabola.end_point.add_dependent(self)
        self._on_point_moved()

    def _on_point_moved(self):
        """Stay below the mid point of the parabola when its end points move."""
        start_x = self.parabola.start_point.get_position().x()
        end_x = self.parabola.end_point.get_position().x()
        self.setPos((start_x + end_x) / 2, self.fixed_y)

    def set_omega(self, omega: float):
        """Flip the arrow when the sign of the rotational speed changes."""
        self.omega = omega
        self.update()

    def boundingRect(self):
        half = self.LENGTH / 2
        return QRectF(-half, -half - 2, 2 * half, self.LENGTH + 30)

    def paint(self, painter: QPainter, _option, _widget):
        if self.omega == 0:
            return
        direction = -1 if self.omega > 0 else 1  # scene y points down
        half = self.LENGTH / 2
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.COLOR, 2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.COLOR))
        tip = QPointF(0, direction * half)
        painter.drawLine(QPointF(0, -direction * half), tip)
        base_y = direction * (half - self.HEAD)
        painter.drawPolygon(
            QPolygonF(
                [tip, QPointF(-self.HEAD / 3, base_y), QPointF(self.HEAD / 3, base_y)]
            )
        )


class RadiusAxis(QGraphicsItem):
    """A vertical radius axis in meters with a live marker following a point.

    The radius is measured upwards from ``origin_y`` (scene y points down) and
    ``scene_per_meter`` converts it to scene units. The axis starts at
    ``min_extent`` and grows as the tracked point is dragged upwards, so there is
    no upper limit. Tick labels and the readout keep a constant size while the
    view is scaled.
    """

    TITLE_OFFSET = 30
    MARGIN = 60

    def __init__(
        self,
        tracked: DraggablePoint,
        axis_x: float,
        origin_y: float,
        scene_per_meter: float,
        tick_step: float,
        min_extent: float,
    ):
        super().__init__()
        self.tracked = tracked
        self.axis_x = axis_x
        self.origin_y = origin_y
        self.scene_per_meter = scene_per_meter
        self.tick_step = tick_step
        self.extent = min_extent  # meters covered by the axis line and ticks
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)

        self.tick_labels: list[QGraphicsSimpleTextItem] = []
        self.title = self._make_label('radius [m]')
        self.readout = self._make_label('')
        self.readout.setBrush(QBrush(QColor(255, 200, 50)))

        tracked.add_dependent(self)
        self._on_point_moved()

    def _make_label(self, text: str) -> QGraphicsSimpleTextItem:
        label = QGraphicsSimpleTextItem(text, self)
        label.setBrush(QBrush(QColor('white')))
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        return label

    def _y(self, radius: float) -> float:
        """Scene y of a radius in meters."""
        return self.origin_y - radius * self.scene_per_meter

    def radius(self) -> float:
        """Radius of the tracked point in meters."""
        return (self.origin_y - self.tracked.get_position().y()) / self.scene_per_meter

    def _grow(self):
        """Extend the axis so it always reaches past the tracked point."""
        needed = self.radius() + self.MARGIN / self.scene_per_meter
        while self.extent < needed:
            self.extent *= 2
        while len(self.tick_labels) <= self.extent / self.tick_step:
            i = len(self.tick_labels)
            label = self._make_label(f'{i * self.tick_step:g}')
            label.setPos(self.axis_x + 6, self._y(i * self.tick_step) - 8)
            self.tick_labels.append(label)
        self.title.setPos(self.axis_x - 20, self._y(self.extent) - self.TITLE_OFFSET)

    def boundingRect(self):
        tracked = self.tracked.get_position()
        top = self._y(self.extent) - self.TITLE_OFFSET - 10
        left = min(self.axis_x, tracked.x()) - 30
        right = max(self.axis_x, tracked.x()) + 120
        bottom = max(self.origin_y, tracked.y()) + 20
        return QRectF(left, top, right - left, bottom - top)

    def paint(self, painter: QPainter, _option, _widget):
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawLine(
            QPointF(self.axis_x, self.origin_y),
            QPointF(self.axis_x, self._y(self.extent)),
        )
        for i in range(len(self.tick_labels)):
            y = self._y(i * self.tick_step)
            painter.drawLine(QPointF(self.axis_x - 4, y), QPointF(self.axis_x + 4, y))

        # Marker line from the axis to the tracked point
        tracked = self.tracked.get_position()
        painter.setPen(QPen(QColor(255, 200, 50), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(self.axis_x, tracked.y()), tracked)

    def _on_point_moved(self):
        """Refresh the marker and readout when the tracked point moves."""
        self.prepareGeometryChange()
        self._grow()
        tracked = self.tracked.get_position()
        self.readout.setText(f'r = {self.radius():.4f} m')
        self.readout.setPos(self.axis_x - 100, tracked.y() - 8)
        self.update()
