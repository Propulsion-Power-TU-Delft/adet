import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QApplication, QGraphicsItem, QGraphicsLineItem


CONTROL_MARKER_RADIUS = 4.0
ANGLE_SNAP_DEG = 10.0
PARABOLA_ANGLE_SNAP_DEG = 5.0
MERIDIONAL_SNAP_DISTANCE = 4.0  # scene units within which the second line snaps


# Point travel per unit of cursor travel
SLOW_DRAG_FACTOR = 0.7
FAST_DRAG_FACTOR = 1.0  # while Shift is held


def _ctrl_held() -> bool:
    return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)


def _shift_held() -> bool:
    return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)


class DraggablePoint(QGraphicsItem):
    """A draggable point in the graphics scene."""

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 5.0,
        parent=None,
        axis_constraint: str | None = None,
        min_x: float | None = None,
        max_x: float | None = None,
        min_y: float | None = None,
        max_y: float | None = None,
    ):
        super().__init__(parent)
        self.radius = radius
        self.axis_constraint = axis_constraint  # None, 'x', or 'y'
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        # The view uses ScrollHandDrag, so override the hand cursor over points
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self._dependents = []
        self._release_listeners = []

    def boundingRect(self):
        return QRectF(
            -self.radius,
            -self.radius,
            2 * self.radius,
            2 * self.radius,
        )

    def paint(self, painter: QPainter, _option, _widget):
        if self.axis_constraint == 'x':
            pen_color = QColor(200, 50, 50)
            brush_color = QColor(255, 150, 100)
        elif self.axis_constraint == 'y':
            pen_color = QColor(50, 200, 50)
            brush_color = QColor(100, 255, 100)
        else:
            pen_color = QColor(50, 50, 200)
            brush_color = QColor(100, 150, 255)

        painter.setPen(QPen(pen_color, 2))
        painter.setBrush(QBrush(brush_color))
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

    def get_position(self) -> QPointF:
        """Get the current position of the point."""
        return self.pos()

    def add_dependent(self, item):
        """Register an item that depends on this point's position."""
        if item not in self._dependents:
            self._dependents.append(item)

    def remove_dependent(self, item):
        """Stop notifying an item that was registered with ``add_dependent``."""
        if item in self._dependents:
            self._dependents.remove(item)

    def mousePressEvent(self, event):
        self._last_scene_pos = event.scenePos()
        self._drag_target = self.pos()
        super().mousePressEvent(event)

    def add_release_listener(self, callback):
        """Call ``callback()`` when the mouse button is released on this point."""
        self._release_listeners.append(callback)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        for callback in self._release_listeners:
            callback()

    def mouseMoveEvent(self, event):
        """Drag the point slowly by default, at full cursor speed while Shift is held."""
        # Move by a fraction of the cursor travel since the last event, so the
        # point can be toggled between slow and fast without jumping
        delta = event.scenePos() - self._last_scene_pos
        self._last_scene_pos = event.scenePos()
        factor = FAST_DRAG_FACTOR if _shift_held() else SLOW_DRAG_FACTOR
        # Accumulate on the unsnapped target: snapping and constraints only modify
        # the position that is applied, so they never eat into the drag travel
        self._drag_target += delta * factor
        self.setPos(self._drag_target)

    def itemChange(self, change, value):
        """Notify dependents when position changes."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            constrained_value = value
            if self.axis_constraint == 'x':
                constrained_value = QPointF(value.x(), self.pos().y())
            elif self.axis_constraint == 'y':
                constrained_value = QPointF(self.pos().x(), value.y())

            x = constrained_value.x()
            y = constrained_value.y()
            if self.min_x is not None:
                x = max(x, self.min_x)
            if self.max_x is not None:
                x = min(x, self.max_x)
            if self.min_y is not None:
                y = max(y, self.min_y)
            if self.max_y is not None:
                y = min(y, self.max_y)
            constrained_value = QPointF(x, y)
            return super().itemChange(change, constrained_value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Notify only after the position is applied, so dependents read the new value
            for dependent in self._dependents:
                dependent._on_point_moved()
        return super().itemChange(change, value)


class SimpleLine(QGraphicsItem):
    """A line connecting two points that updates when they move."""

    def __init__(self, point1: DraggablePoint, point2: DraggablePoint, parent=None):
        super().__init__(parent)
        self.point1 = point1
        self.point2 = point2
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)
        # Register as dependent on both control points
        self.point1.add_dependent(self)
        self.point2.add_dependent(self)

    def boundingRect(self):
        p1 = self.point1.get_position()
        p2 = self.point2.get_position()
        min_x = min(p1.x(), p2.x()) - 2
        min_y = min(p1.y(), p2.y()) - 2
        max_x = max(p1.x(), p2.x()) + 2
        max_y = max(p1.y(), p2.y()) + 2
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter: QPainter, _option, _widget):
        painter.setPen(QPen(QColor(200, 100, 100), 2))
        p1 = self.point1.get_position()
        p2 = self.point2.get_position()
        painter.drawLine(p1, p2)

    def _on_point_moved(self):
        """Called when a control point moves to invalidate and redraw."""
        self.prepareGeometryChange()
        self.update()


class ParabolicLine(QGraphicsItem):
    """A parabolic line controlled by three points using quadratic Bezier curve."""

    def __init__(
        self,
        start_point: DraggablePoint,
        control_point: DraggablePoint,
        end_point: DraggablePoint,
        parent=None,
        show_control_polygon: bool = False,
    ):
        super().__init__(parent)
        self.color = QColor(255, 255, 255)
        self.show_control_polygon = show_control_polygon
        self.start_point = start_point
        self.control_point = control_point
        self.end_point = end_point
        self.segments = 300
        self._updating_control = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)

        # Register as dependent on all control points
        self.start_point.add_dependent(self)
        self.control_point.add_dependent(self)
        self.end_point.add_dependent(self)

        # Constrain start point to vertical movement only
        self.start_point.axis_constraint = 'y'

        # Constrain end point x to be at least the start point's x
        self.end_point.min_x = self.start_point.get_position().x()

        # Override control_point's itemChange to enforce vertical movement
        # and keep its x-coordinate at the midpoint between start and end
        original_control_itemChange = self.control_point.itemChange

        def control_constrained_itemChange(change, value):
            if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
                # Apply y-constraint: keep current y, use new position's y
                current_pos = self.control_point.pos()
                y_constrained = QPointF(current_pos.x(), value.y())

                # Apply midpoint x-constraint
                start_x = self.start_point.get_position().x()
                end_x = self.end_point.get_position().x()
                midpoint_x = (start_x + end_x) / 2
                final_value = QPointF(midpoint_x, y_constrained.y())

                # Only notify dependents if not updating from _on_point_moved
                if not self._updating_control:
                    for dependent in self.control_point._dependents:
                        dependent._on_point_moved()

                # Call parent's itemChange
                return QGraphicsItem.itemChange(self.control_point, change, final_value)

            return original_control_itemChange(change, value)

        self.control_point.itemChange = control_constrained_itemChange  # type: ignore

        # Snap the tangent angles at both ends while Ctrl is held
        original_start_itemChange = self.start_point.itemChange
        original_end_itemChange = self.end_point.itemChange

        def start_snapped_itemChange(change, value):
            if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
                value = self._snap_start(value)
            return original_start_itemChange(change, value)

        def end_snapped_itemChange(change, value):
            if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
                value = self._snap_end(value)
            return original_end_itemChange(change, value)

        self.start_point.itemChange = start_snapped_itemChange  # type: ignore
        self.end_point.itemChange = end_snapped_itemChange  # type: ignore

    def _snap_start(self, value: QPointF) -> QPointF:
        """Snap the inlet angle to PARABOLA_ANGLE_SNAP_DEG steps while Ctrl is held.

        The start point only moves vertically, so the snapped angle fixes its y.
        """
        movable = (
            self.start_point.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable
        )
        if not movable or not _ctrl_held():
            return value
        start_x = self.start_point.pos().x()
        control = self.control_point.get_position()
        dx = control.x() - start_x
        if dx <= 0:
            return value
        step = math.radians(PARABOLA_ANGLE_SNAP_DEG)
        angle = round(math.atan2(control.y() - value.y(), dx) / step) * step
        if abs(math.cos(angle)) < 1e-9:
            return value
        return QPointF(start_x, control.y() - dx * math.tan(angle))

    def _snap_end(self, value: QPointF) -> QPointF:
        """Snap the outlet angle to PARABOLA_ANGLE_SNAP_DEG steps while Ctrl is held.

        The control point follows the end point to the midpoint in x, so the angle
        is measured from that future control position.
        """
        movable = self.end_point.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable
        if not movable or not _ctrl_held():
            return value
        start_x = self.start_point.get_position().x()
        control_y = self.control_point.get_position().y()
        dx = (value.x() - start_x) / 2
        dy = value.y() - control_y
        radius = math.hypot(dx, dy)
        if radius == 0:
            return value
        step = math.radians(PARABOLA_ANGLE_SNAP_DEG)
        angle = round(math.atan2(dy, dx) / step) * step
        return QPointF(
            start_x + 2 * radius * math.cos(angle),
            control_y + radius * math.sin(angle),
        )

    def _get_bezier_point(self, t: float) -> QPointF:
        """Calculate a point on the quadratic Bezier curve at parameter t (0-1)."""
        p0 = self.start_point.get_position()
        p1 = self.control_point.get_position()
        p2 = self.end_point.get_position()

        x = (1 - t) ** 2 * p0.x() + 2 * (1 - t) * t * p1.x() + t**2 * p2.x()
        y = (1 - t) ** 2 * p0.y() + 2 * (1 - t) * t * p1.y() + t**2 * p2.y()
        return QPointF(x, y)

    @staticmethod
    def _direction_angle(origin: QPointF, target: QPointF) -> float:
        """Angle in degrees of the vector origin -> target, in scene coordinates."""
        return math.degrees(
            math.atan2(target.y() - origin.y(), target.x() - origin.x())
        )

    @property
    def inlet_angle(self) -> float:
        """Tangent angle (deg) at the start of the parabola, measured from the x axis."""
        return self._direction_angle(
            self.start_point.get_position(), self.control_point.get_position()
        )

    @property
    def outlet_angle(self) -> float:
        """Tangent angle (deg) at the end of the parabola, measured from the x axis."""
        return self._direction_angle(
            self.control_point.get_position(), self.end_point.get_position()
        )

    def boundingRect(self):
        points = [
            self._get_bezier_point(i / self.segments) for i in range(self.segments + 1)
        ]
        if self.show_control_polygon:
            points += self._control_polygon()
        if not points:
            return QRectF()
        min_x = min(p.x() for p in points) - 2
        min_y = min(p.y() for p in points) - 2
        max_x = max(p.x() for p in points) + 2
        max_y = max(p.y() for p in points) + 2
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def _control_polygon(self) -> list[QPointF]:
        return [
            self.start_point.get_position(),
            self.control_point.get_position(),
            self.end_point.get_position(),
        ]

    def paint(self, painter: QPainter, _option, _widget):
        if self.show_control_polygon:
            polygon = self._control_polygon()
            painter.setPen(QPen(QColor(128, 128, 128), 1, Qt.PenStyle.DashLine))
            for a, b in zip(polygon, polygon[1:]):
                painter.drawLine(a, b)
        painter.setPen(QPen(self.color, 2))
        points = [
            self._get_bezier_point(i / self.segments) for i in range(self.segments + 1)
        ]
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

    def _on_point_moved(self):
        """Called when any control point moves to invalidate and redraw."""
        # Update control point's x to be the midpoint between start and end
        self._updating_control = True
        try:
            start_x = self.start_point.get_position().x()
            end_x = self.end_point.get_position().x()
            midpoint_x = (start_x + end_x) / 2
            current_y = self.control_point.get_position().y()
            self.control_point.setPos(midpoint_x, current_y)
        finally:
            self._updating_control = False

        self.prepareGeometryChange()
        self.update()

    def update_geometry(self):
        """Public method to update parabolic line geometry."""
        self._on_point_moved()


class CubicBezierLine(QGraphicsItem):
    """A cubic Bezier curve controlled by four points."""

    def __init__(
        self,
        start_point,
        control_point1,
        control_point2,
        end_point,
        parent=None,
        show_control_polygon: bool = False,
        color: QColor | None = None,
    ):
        super().__init__(parent)
        self.show_control_polygon = show_control_polygon
        self.color = color if color is not None else QColor(200, 150, 100)
        self.start_point = start_point
        self.control_point1 = control_point1
        self.control_point2 = control_point2
        self.end_point = end_point
        self.segments = 300
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)

        # Register as dependent on all control points (if they support it)
        if hasattr(self.start_point, 'add_dependent'):
            self.start_point.add_dependent(self)
        if hasattr(self.control_point1, 'add_dependent'):
            self.control_point1.add_dependent(self)
        if hasattr(self.control_point2, 'add_dependent'):
            self.control_point2.add_dependent(self)
        if hasattr(self.end_point, 'add_dependent'):
            self.end_point.add_dependent(self)

    def _get_bezier_point(self, t: float) -> QPointF:
        """Calculate a point on the cubic Bezier curve at parameter t (0-1)."""
        p0 = self.start_point.get_position()
        p1 = self.control_point1.get_position()
        p2 = self.control_point2.get_position()
        p3 = self.end_point.get_position()

        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        t2 = t * t
        t3 = t2 * t

        x = mt3 * p0.x() + 3 * mt2 * t * p1.x() + 3 * mt * t2 * p2.x() + t3 * p3.x()
        y = mt3 * p0.y() + 3 * mt2 * t * p1.y() + 3 * mt * t2 * p2.y() + t3 * p3.y()
        return QPointF(x, y)

    def _control_polygon(self) -> list[QPointF]:
        return [
            self.start_point.get_position(),
            self.control_point1.get_position(),
            self.control_point2.get_position(),
            self.end_point.get_position(),
        ]

    def boundingRect(self):
        points = [
            self._get_bezier_point(i / self.segments) for i in range(self.segments + 1)
        ]
        if self.show_control_polygon:
            points += self._control_polygon()
        min_x = min(p.x() for p in points) - CONTROL_MARKER_RADIUS - 2
        min_y = min(p.y() for p in points) - CONTROL_MARKER_RADIUS - 2
        max_x = max(p.x() for p in points) + CONTROL_MARKER_RADIUS + 2
        max_y = max(p.y() for p in points) + CONTROL_MARKER_RADIUS + 2
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter: QPainter, _option, _widget):
        if self.show_control_polygon:
            polygon = self._control_polygon()
            painter.setPen(QPen(QColor(128, 128, 128), 1, Qt.PenStyle.DashLine))
            for a, b in zip(polygon, polygon[1:]):
                painter.drawLine(a, b)
            # Inner control points (end points are already draggable points)
            painter.setPen(QPen(QColor(90, 90, 90), 1))
            painter.setBrush(QBrush(QColor(200, 200, 200)))
            for point in polygon[1:3]:
                painter.drawEllipse(point, CONTROL_MARKER_RADIUS, CONTROL_MARKER_RADIUS)

        painter.setPen(QPen(self.color, 2))
        points = [
            self._get_bezier_point(i / self.segments) for i in range(self.segments + 1)
        ]
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

    def _on_point_moved(self):
        """Called when any control point moves to invalidate and redraw."""
        self.prepareGeometryChange()
        self.update()


class SymmetricLine(QGraphicsItem):
    """A line controlled by a center point with symmetric endpoints.

    The two endpoints are always symmetric with respect to the center.
    Dragging an endpoint changes length and rotation while maintaining symmetry.
    Dragging the center translates the entire line.
    """

    def __init__(
        self,
        center_point: DraggablePoint,
        end_point: DraggablePoint,
        parent=None,
        color: QColor | None = None,
    ):
        super().__init__(parent)
        self.color = color if color is not None else QColor(100, 150, 200)
        self.center = center_point
        self.end1 = end_point
        self._updating_endpoints = False

        # Create the symmetric endpoint
        center_pos = self.center.get_position()
        end1_pos = self.end1.get_position()
        # end2 is symmetric: center + (center - end1)
        end2_x = 2 * center_pos.x() - end1_pos.x()
        end2_y = 2 * center_pos.y() - end1_pos.y()
        self.end2 = DraggablePoint(
            end2_x, end2_y, radius=end_point.radius, parent=parent
        )

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)

        # Register as dependent on all points
        self.center.add_dependent(self)
        self.end1.add_dependent(self)
        self.end2.add_dependent(self)

        # Override itemChange handlers to maintain symmetry
        original_end1_itemChange = self.end1.itemChange
        original_end2_itemChange = self.end2.itemChange
        original_center_itemChange = self.center.itemChange

        def end1_constrained_itemChange(change, value):
            if (
                change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
                and not self._updating_endpoints
            ):
                value = self._snap_angle(value)
                self._updating_endpoints = True
                try:
                    # Update end2 to be symmetric
                    center_pos = self.center.get_position()
                    end2_new = QPointF(
                        2 * center_pos.x() - value.x(), 2 * center_pos.y() - value.y()
                    )
                    self.end2.setPos(end2_new)
                finally:
                    self._updating_endpoints = False

            # Call original itemChange
            return original_end1_itemChange(change, value)

        def end2_constrained_itemChange(change, value):
            if (
                change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
                and not self._updating_endpoints
            ):
                value = self._snap_angle(value)
                self._updating_endpoints = True
                try:
                    # Update end1 to be symmetric
                    center_pos = self.center.get_position()
                    end1_new = QPointF(
                        2 * center_pos.x() - value.x(), 2 * center_pos.y() - value.y()
                    )
                    self.end1.setPos(end1_new)
                finally:
                    self._updating_endpoints = False

            # Call original itemChange
            return original_end2_itemChange(change, value)

        def center_constrained_itemChange(change, value):
            if (
                change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
                and not self._updating_endpoints
            ):
                self._updating_endpoints = True
                try:
                    # Apply axis constraint if center has one
                    constrained_value = value
                    if hasattr(self.center, 'axis_constraint'):
                        if self.center.axis_constraint == 'x':
                            constrained_value = QPointF(
                                value.x(), self.center.pos().y()
                            )
                        elif self.center.axis_constraint == 'y':
                            constrained_value = QPointF(
                                self.center.pos().x(), value.y()
                            )

                    # Clamp to the center's bounds so the endpoints follow the
                    # actual (clamped) displacement
                    cx, cy = constrained_value.x(), constrained_value.y()
                    if self.center.min_x is not None:
                        cx = max(cx, self.center.min_x)
                    if self.center.max_x is not None:
                        cx = min(cx, self.center.max_x)
                    if self.center.min_y is not None:
                        cy = max(cy, self.center.min_y)
                    if self.center.max_y is not None:
                        cy = min(cy, self.center.max_y)
                    constrained_value = QPointF(cx, cy)

                    # Calculate displacement based on constrained value
                    old_center = self.center.pos()
                    dx = constrained_value.x() - old_center.x()
                    dy = constrained_value.y() - old_center.y()

                    # Move both endpoints by the same displacement
                    end1_pos = self.end1.get_position()
                    end2_pos = self.end2.get_position()
                    self.end1.setPos(end1_pos.x() + dx, end1_pos.y() + dy)
                    self.end2.setPos(end2_pos.x() + dx, end2_pos.y() + dy)
                finally:
                    self._updating_endpoints = False

            # Call original itemChange
            return original_center_itemChange(change, value)

        self.end1.itemChange = end1_constrained_itemChange  # type: ignore
        self.end2.itemChange = end2_constrained_itemChange  # type: ignore
        self.center.itemChange = center_constrained_itemChange  # type: ignore

    def _snap_angle(self, value: QPointF) -> QPointF:
        """Snap an endpoint position to ANGLE_SNAP_DEG steps while Ctrl is held."""
        if not _ctrl_held():
            return value
        center_pos = self.center.get_position()
        dx = value.x() - center_pos.x()
        dy = value.y() - center_pos.y()
        radius = math.hypot(dx, dy)
        if radius == 0:
            return value
        step = math.radians(ANGLE_SNAP_DEG)
        angle = round(math.atan2(dy, dx) / step) * step
        return QPointF(
            center_pos.x() + radius * math.cos(angle),
            center_pos.y() + radius * math.sin(angle),
        )

    def boundingRect(self):
        p1 = self.end1.get_position()
        p2 = self.end2.get_position()
        pc = self.center.get_position()

        min_x = min(p1.x(), p2.x(), pc.x()) - 2
        min_y = min(p1.y(), p2.y(), pc.y()) - 2
        max_x = max(p1.x(), p2.x(), pc.x()) + 2
        max_y = max(p1.y(), p2.y(), pc.y()) + 2

        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter: QPainter, _option, _widget):
        painter.setPen(QPen(self.color, 2))
        p1 = self.end1.get_position()
        p2 = self.end2.get_position()
        painter.drawLine(p1, p2)

        # Draw center point indicator
        painter.setPen(QPen(QColor(200, 200, 50), 1))
        pc = self.center.get_position()
        painter.drawEllipse(pc, 2, 2)

    def _on_point_moved(self):
        """Called when any control point moves to invalidate and redraw."""
        self.prepareGeometryChange()
        self.update()


class StaticPoint:
    """A point that doesn't interact with the scene, just holds position."""

    def __init__(self, x: float, y: float):
        self._pos = QPointF(x, y)

    def get_position(self) -> QPointF:
        return self._pos

    def setPos(self, x: float, y: float):
        self._pos = QPointF(x, y)


class AlignedPoints:
    """Keeps two points aligned along the x or y axis.

    ``axis='x'`` places both points on a line parallel to the x axis (same y).
    ``axis='y'`` places both points on a line parallel to the y axis (same x).
    The constraint works in both directions: whichever point is moved drags the
    other one along. If both differ from their last values, the leader wins.
    When aligning at creation and when the leader's own limits stop it, the
    follower adapts to the leader. The alignment overrides any
    ``axis_constraint`` of the follower.
    """

    def __init__(self, leader: DraggablePoint, follower: DraggablePoint, axis: str):
        if axis not in ('x', 'y'):
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        self.leader = leader
        self.follower = follower
        self.axis = axis
        self._syncing = False
        self.leader.add_dependent(self)
        self.follower.add_dependent(self)
        self._sync_follower()
        self._last_leader = self._value(self.leader)
        self._last_follower = self._value(self.follower)

    def _value(self, point) -> float:
        """Coordinate that must match between the two points."""
        pos = point.get_position()
        return pos.y() if self.axis == 'x' else pos.x()

    def _set_value(self, point, value: float):
        pos = point.get_position()
        if self.axis == 'x':
            point.setPos(pos.x(), value)
        else:
            point.setPos(value, pos.y())

    def _sync_follower(self):
        """Move the follower to the leader's aligned coordinate."""
        # The alignment must win over the follower's own axis constraint
        constraint = self.follower.axis_constraint
        self.follower.axis_constraint = None
        try:
            self._set_value(self.follower, self._value(self.leader))
        finally:
            self.follower.axis_constraint = constraint

    def _on_point_moved(self):
        """Keep the points aligned, following whichever one moved."""
        if self._syncing:
            return
        self._syncing = True
        try:
            leader_moved = self._value(self.leader) != self._last_leader
            follower_moved = self._value(self.follower) != self._last_follower
            if follower_moved and not leader_moved:
                self._set_value(self.leader, self._value(self.follower))
            # The leader may have been clamped, so the follower adapts to it
            self._sync_follower()
            self._last_leader = self._value(self.leader)
            self._last_follower = self._value(self.follower)
        finally:
            self._syncing = False


class PerpendicularPoints(QGraphicsItem):
    """Keeps three points perpendicular at the middle point.

    The middle point (point2) is the junction of two line segments:
    point1 → point2 and point2 → point3, which stay perpendicular.
    Only point1 and point2 are draggable; point3 is calculated.
    """

    def __init__(
        self,
        point1,
        point2,
        parent=None,
        point3_distance: float = 50.0,
        distance_to=None,
        distance_fraction: float = 1 / 3,
    ):
        super().__init__(parent)
        self.point1 = point1
        self.point2 = point2
        self.point3_distance = point3_distance
        # If given, |point3_distance| is replaced by distance_fraction times the
        # distance from point2 to this point; the sign still picks the side
        self.distance_to = distance_to
        self.distance_fraction = distance_fraction

        # point3 is updated in place so that other items holding a reference to it
        # (e.g. a spline control point) always see its current position
        self.point3 = StaticPoint(0, 0)
        self._update_point3()
        self._dependents = []

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)

        # Only register as dependent if the point supports it
        if hasattr(self.point1, 'add_dependent'):
            self.point1.add_dependent(self)
        if hasattr(self.point2, 'add_dependent'):
            self.point2.add_dependent(self)
        if hasattr(self.distance_to, 'add_dependent'):
            self.distance_to.add_dependent(self)

    def _current_distance(self) -> float:
        """Signed distance from point2 to point3."""
        if self.distance_to is None:
            return self.point3_distance
        p2 = self.point2.get_position()
        target = self.distance_to.get_position()
        length = math.hypot(target.x() - p2.x(), target.y() - p2.y())
        return math.copysign(self.distance_fraction * length, self.point3_distance)

    def _update_point3(self):
        """Update point3 based on perpendicularity and its distance from point2."""
        p1 = self.point1.get_position()
        p2 = self.point2.get_position()
        distance = self._current_distance()

        v1_dx = p2.x() - p1.x()
        v1_dy = p2.y() - p1.y()

        perp_dx = -v1_dy
        perp_dy = v1_dx

        perp_len = math.sqrt(perp_dx**2 + perp_dy**2)
        if perp_len == 0:
            self.point3.setPos(p2.x(), p2.y())
            return

        perp_dx_norm = (perp_dx / perp_len) * distance
        perp_dy_norm = (perp_dy / perp_len) * distance

        self.point3.setPos(p2.x() + perp_dx_norm, p2.y() + perp_dy_norm)

    def boundingRect(self):
        p1 = self.point1.get_position()
        p2 = self.point2.get_position()
        p3 = self.point3.get_position()

        min_x = min(p1.x(), p2.x(), p3.x()) - 2
        min_y = min(p1.y(), p2.y(), p3.y()) - 2
        max_x = max(p1.x(), p2.x(), p3.x()) + 2
        max_y = max(p1.y(), p2.y(), p3.y()) + 2

        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter: QPainter, _option, _widget):
        painter.setPen(QPen(QColor(100, 150, 200), 2))
        p1 = self.point1.get_position()
        p2 = self.point2.get_position()
        p3 = self.point3.get_position()
        painter.drawLine(p1, p2)
        painter.drawLine(p2, p3)

    def add_dependent(self, item):
        """Register an item that depends on this constraint's point3."""
        if item not in self._dependents:
            self._dependents.append(item)

    def _on_point_moved(self):
        """Update point3 when any draggable point moves."""
        self._update_point3()
        for dependent in self._dependents:
            dependent._on_point_moved()
        self.prepareGeometryChange()
        self.update()


class GapFollower:
    """Keeps a line shifted along the wall direction from a source line.

    The shift has length ``gap`` along the meridional direction, i.e. normal to the
    source line, so the endwalls of both lines meet as if the gap were not there. The
    followed line has the same length and tilt as the source one.
    """

    def __init__(
        self,
        source_center: DraggablePoint,
        source_end: DraggablePoint,
        center: DraggablePoint,
        end: DraggablePoint,
        gap: float,
    ):
        self.source_center = source_center
        self.source_end = source_end
        self.center = center
        self.end = end
        self.gap = gap
        self._syncing = False
        source_center.add_dependent(self)
        source_end.add_dependent(self)

    @staticmethod
    def offset(
        source_center: DraggablePoint, source_end: DraggablePoint, gap: float
    ) -> QPointF:
        """Shift of the source line to its gapped copy."""
        dx = source_end.pos().x() - source_center.pos().x()
        dy_up = source_center.pos().y() - source_end.pos().y()
        length = math.hypot(dx, dy_up)
        if length < 1e-9:
            return QPointF(gap, 0.0)
        # The gap is measured along the meridional direction (normal to the line).
        # The line is directed from its center to its end, as built by _line_points,
        # so the normal (cos(angle), sin(angle) in scene coordinates) turns
        # continuously through 90 deg instead of flipping its radial sign there.
        # Working with the normal directly, instead of tan(angle), stays finite when
        # the line is horizontal. Scene y points down
        return QPointF(gap * dy_up / length, gap * dx / length)

    def detach(self):
        self.source_center.remove_dependent(self)
        self.source_end.remove_dependent(self)

    def _on_point_moved(self):
        if self._syncing:
            return
        self._syncing = True
        try:
            shift = self.offset(self.source_center, self.source_end, self.gap)
            # The endpoints follow the center, then are set to the source ones
            self.center.setPos(self.source_center.pos() + shift)
            self.end.setPos(self.source_end.pos() + shift)
        finally:
            self._syncing = False


class MeridionalFill(QGraphicsItem):
    """Translucent fill of the region enclosed by the two lines and two splines."""

    OPACITY = 0.2

    def __init__(
        self,
        spline_a: CubicBezierLine,
        spline_b: CubicBezierLine,
        color: QColor,
        parent=None,
    ):
        super().__init__(parent)
        self.spline_a = spline_a
        self.spline_b = spline_b
        self.color = color
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)

    def _path(self) -> QPainterPath:
        a, b = self.spline_a, self.spline_b
        path = QPainterPath(a.start_point.get_position())
        path.cubicTo(
            a.control_point1.get_position(),
            a.control_point2.get_position(),
            a.end_point.get_position(),
        )
        path.lineTo(b.end_point.get_position())
        path.cubicTo(
            b.control_point2.get_position(),
            b.control_point1.get_position(),
            b.start_point.get_position(),
        )
        path.closeSubpath()
        return path

    def boundingRect(self):
        return self._path().controlPointRect().adjusted(-2, -2, 2, 2)

    def paint(self, painter: QPainter, _option, _widget):
        fill = QColor(self.color)
        fill.setAlphaF(self.OPACITY)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill))
        painter.drawPath(self._path())

    def _on_point_moved(self):
        """Called when a constraint driving the region boundary changes."""
        self.prepareGeometryChange()
        self.update()


class SnapHighlight(QGraphicsLineItem):
    """Temporary red dotted line drawn on top of everything while a snap is active."""

    def __init__(self):
        super().__init__()
        pen = QPen(QColor(255, 40, 40), 2, Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(1e9)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setVisible(False)

    def show_between(self, scene, start: QPointF, end: QPointF):
        if self.scene() is not scene:
            scene.addItem(self)
        self.setLine(start.x(), start.y(), end.x(), end.y())
        self.setVisible(True)

    def clear(self):
        self.setVisible(False)


class MeridionalProfile:
    """A meridional profile made of two symmetric lines joined by two splines.

    Each symmetric line has a draggable center and two mirrored endpoints. Each
    endpoint has a perpendicular constraint whose calculated point acts as a
    control point of a cubic Bezier spline, so the splines leave both lines
    at a right angle. The first spline joins the ``end1`` points of the lines
    and the second joins the ``end2`` points.

    With ``previous`` and a ``gap`` the first line is a copy of the last line of
    ``previous``, shifted by ``gap`` along the meridional direction. It follows
    ``previous`` and cannot be dragged. Without a gap the line is shared.
    """

    def __init__(
        self,
        center1: tuple[float, float] = (150, 600),
        end1: tuple[float, float] = (150, 400),
        center2: tuple[float, float] = (550, 400),
        end2: tuple[float, float] = (550, 200),
        control_fraction: float = 1 / 3,
        center1_axis_constraint: str | None = 'y',
        line_color: QColor | None = None,
        previous: 'MeridionalProfile | None' = None,
        gap: float = 0.0,
    ):
        if line_color is None:
            line_color = QColor(255, 255, 255)

        # First line; when following another profile without a gap, its inlet station
        # is the outlet station of that profile (same points, so they always coincide)
        self._shares_line1 = previous is not None and gap == 0
        self.gap_follower: GapFollower | None = None
        if previous is None or gap != 0:
            if previous is not None:
                shift = GapFollower.offset(previous.center2, previous.end2, gap)
                start = previous.center2.pos() + shift
                tip = previous.end2.pos() + shift
                center1, end1 = (start.x(), start.y()), (tip.x(), tip.y())
                center1_axis_constraint = None  # it follows the previous profile
            self.center1 = DraggablePoint(
                *center1, axis_constraint=center1_axis_constraint
            )
            self.end1 = DraggablePoint(*end1)
            self.line1 = SymmetricLine(self.center1, self.end1, color=line_color)
            if previous is not None:
                for point in (self.center1, self.end1, self.line1.end2):
                    point.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                self.gap_follower = GapFollower(
                    previous.center2, previous.end2, self.center1, self.end1, gap
                )
        else:
            self.center1 = previous.center2
            self.end1 = previous.end2
            self.line1 = previous.line2

        # Second line
        # The second center cannot go left of the first center
        self.center2 = DraggablePoint(*center2, min_x=self.center1.pos().x())
        self.end2 = DraggablePoint(*end2)
        self.line2 = SymmetricLine(self.center2, self.end2, color=line_color)

        # Dragging the second center snaps the line to the first one
        original_center2_itemChange = self.center2.itemChange

        def center2_snapped_itemChange(change, value):
            if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
                scene = self.center2.scene()
                if (
                    _ctrl_held()
                    and scene is not None
                    and scene.mouseGrabberItem() is self.center2
                ):
                    # Ctrl while dragging restricts the move to the y direction
                    value = QPointF(self.center2.pos().x(), value.y())
                value = self._snap_center2(value)
            return original_center2_itemChange(change, value)

        self.center2.itemChange = center2_snapped_itemChange  # type: ignore

        # Red dotted line showing what a snap aligns to, for the length of the drag
        self.snap_highlight = SnapHighlight()
        self.center2.add_release_listener(self.snap_highlight.clear)
        self.line2.end1.add_release_listener(self.snap_highlight.clear)
        self.line2.end2.add_release_listener(self.snap_highlight.clear)

        # Dragging an end of the second line snaps its spline straight as well
        for end, ref, mirror_ref in (
            (self.line2.end1, self.line1.end1, self.line1.end2),
            (self.line2.end2, self.line1.end2, self.line1.end1),
        ):
            self._install_end_snap(end, ref, mirror_ref)

        # Perpendicular constraints; the sign flips the side of the control point.
        # Each control point sits at control_fraction of the distance between the
        # spline's endpoints away from its own endpoint.
        self.perp1_a = PerpendicularPoints(
            self.center1,
            self.line1.end1,
            point3_distance=1,
            distance_to=self.line2.end1,
            distance_fraction=control_fraction,
        )
        self.perp1_b = PerpendicularPoints(
            self.center1,
            self.line1.end2,
            point3_distance=-1,
            distance_to=self.line2.end2,
            distance_fraction=control_fraction,
        )
        self.perp2_a = PerpendicularPoints(
            self.center2,
            self.line2.end1,
            point3_distance=-1,
            distance_to=self.line1.end1,
            distance_fraction=control_fraction,
        )
        self.perp2_b = PerpendicularPoints(
            self.center2,
            self.line2.end2,
            point3_distance=1,
            distance_to=self.line1.end2,
            distance_fraction=control_fraction,
        )

        # Splines between matching endpoints of the two lines
        self.spline_a = CubicBezierLine(
            self.line1.end1,
            self.perp1_a.point3,
            self.perp2_a.point3,
            self.line2.end1,
            show_control_polygon=True,
            color=line_color,
        )
        self.perp1_a.add_dependent(self.spline_a)
        self.perp2_a.add_dependent(self.spline_a)

        self.spline_b = CubicBezierLine(
            self.line1.end2,
            self.perp1_b.point3,
            self.perp2_b.point3,
            self.line2.end2,
            show_control_polygon=True,
            color=line_color,
        )
        self.perp1_b.add_dependent(self.spline_b)
        self.perp2_b.add_dependent(self.spline_b)

        # The fill repaints whenever a constraint (which covers all four line ends
        # and the spline control points) updates
        self.fill = MeridionalFill(self.spline_a, self.spline_b, line_color)
        for perp in self.perpendiculars:
            perp.add_dependent(self.fill)

        # The constraints still drive the control points when hidden
        for perp in self.perpendiculars:
            perp.setVisible(False)

        # Dragging the first center translates the whole profile vertically; a shared
        # station is moved on its own
        self._last_center1_y = self.center1.pos().y()
        # The profiles that slide along with the first one (all later rows)
        self.root: MeridionalProfile = self if previous is None else previous.root
        self.chained: list[MeridionalProfile] = []
        if previous is None:
            self.center1.add_dependent(self)
        else:
            self.root.chained.append(self)

    def _snap_center2(self, value: QPointF) -> QPointF:
        """Snap the dragged second center vertically to the first line.

        The second line moves so that either its center or one of its ends is level
        with the matching point of the first line, i.e. their offset is perpendicular
        to the first line. For the ends this makes the spline between them straight.
        Only a drag by the mouse snaps, not the profile following the first center.
        """
        scene = self.center2.scene()
        if scene is None or scene.mouseGrabberItem() is not self.center2:
            return value
        self.snap_highlight.clear()
        tilt = self.line1.end1.pos() - self.center1.pos()
        if tilt.manhattanLength() < 1e-9:
            return value
        # Shift along y, or along x when the first line is closer to horizontal
        # (radial), where a vertical shift cannot make the offset perpendicular
        shift_x = abs(tilt.x()) > abs(tilt.y())
        axis_tilt = tilt.x() if shift_x else tilt.y()
        pairs = (
            (self.center2, self.center1),
            (self.line2.end1, self.line1.end1),
            (self.line2.end2, self.line1.end2),
        )
        best: float | None = None
        best_line: tuple[QPointF, QPointF] | None = None
        for point, reference in pairs:
            # Offset of the point from the dragged center, unchanged by the drag
            offset = point.pos() - self.center2.pos()
            gap = value + offset - reference.pos()
            # Shift that makes the gap perpendicular to the first line
            shift = -(gap.x() * tilt.x() + gap.y() * tilt.y()) / axis_tilt
            if abs(shift) <= MERIDIONAL_SNAP_DISTANCE and (
                best is None or abs(shift) < abs(best)
            ):
                best = shift
                step = QPointF(shift, 0) if shift_x else QPointF(0, shift)
                best_line = (reference.pos(), value + offset + step)
        if best is None or best_line is None:
            return value
        self.snap_highlight.show_between(scene, *best_line)
        if shift_x:
            return QPointF(value.x() + best, value.y())
        return QPointF(value.x(), value.y() + best)

    def _install_end_snap(
        self, end: DraggablePoint, reference: DraggablePoint, mirror_reference
    ):
        """Snap ``end`` of the second line, when dragged, to a straight spline.

        ``reference`` is the end of the first line joined to ``end`` by a spline and
        ``mirror_reference`` the one joined to the mirrored end of the second line.
        """
        original_itemChange = end.itemChange

        def end_snapped_itemChange(change, value):
            if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
                value = self._snap_end(end, reference, mirror_reference, value)
            return original_itemChange(change, value)

        end.itemChange = end_snapped_itemChange  # type: ignore

    def _snap_end(
        self,
        end: DraggablePoint,
        reference: DraggablePoint,
        mirror_reference: DraggablePoint,
        value: QPointF,
    ) -> QPointF:
        """Move the dragged end so that a spline of the second line becomes straight.

        A spline is straight when the offset between its ends is perpendicular to the
        first line. Either the dragged end or its mirror about the second center can
        be levelled with its counterpart; the shift is along the first line.
        While Ctrl is held the end keeps the angle of the line snapped to its steps,
        so it is only shifted along that direction.
        """
        scene = end.scene()
        if scene is None or scene.mouseGrabberItem() is not end:
            return value
        self.snap_highlight.clear()
        center = self.center2.pos()
        direction: QPointF | None = None
        if _ctrl_held():
            value = self.line2._snap_angle(value)
            direction = value - center
            length = math.hypot(direction.x(), direction.y())
            if length == 0:
                return value
            direction = direction / length
        tilt = self.line1.end1.pos() - self.center1.pos()
        norm2 = tilt.x() ** 2 + tilt.y() ** 2
        if norm2 < 1e-18:
            return value
        # Unit shift of the dragged end that changes the gap along the first line by 1
        if direction is None:
            step = tilt / norm2
        else:
            across = direction.x() * tilt.x() + direction.y() * tilt.y()
            if abs(across) < 1e-9 * math.sqrt(norm2):
                return value  # the end moves perpendicular to the first line
            step = direction / across
        mirror = QPointF(2 * center.x() - value.x(), 2 * center.y() - value.y())
        best: QPointF | None = None
        best_line: tuple[QPointF, QPointF] | None = None
        best_size = MERIDIONAL_SNAP_DISTANCE
        for gap, sign, ref, moved in (
            (value - reference.pos(), -1, reference, value),
            (mirror - mirror_reference.pos(), 1, mirror_reference, mirror),
        ):
            along = gap.x() * tilt.x() + gap.y() * tilt.y()
            shift = sign * along * step
            size = math.hypot(shift.x(), shift.y())
            if size <= best_size:
                best_size = size
                best = shift
                # The mirrored end moves opposite to the dragged one
                best_line = (ref.pos(), moved - sign * best)
        if best is None or best_line is None:
            return value
        self.snap_highlight.show_between(scene, *best_line)
        return value + best

    def set_color(self, color: QColor):
        """Draw the lines and splines of the profile in ``color``.

        A first line shared with the previous profile keeps the color of that profile.
        """
        items = [self.line2, self.spline_a, self.spline_b, self.fill]
        if not self._shares_line1:
            items.append(self.line1)
        for item in items:
            item.color = color
            item.update()

    def _on_point_moved(self):
        """Translate all rows by the vertical displacement of the first center.

        The first line follows its own center; the second center drags its
        endpoints, and the constraints and splines update through their dependencies.
        The profiles of the later rows slide along, so the whole machine moves.
        """
        dy = self.center1.pos().y() - self._last_center1_y
        if dy == 0:
            return
        self._last_center1_y += dy
        for profile in (self, *self.chained):
            pos = profile.center2.pos()
            profile.center2.setPos(pos.x(), pos.y() + dy)

    @property
    def perpendiculars(self) -> tuple[PerpendicularPoints, ...]:
        return self.perp1_a, self.perp1_b, self.perp2_a, self.perp2_b

    @property
    def items(self) -> list[QGraphicsItem]:
        """All graphics items of the profile, in the order they should be added."""
        # The shared inlet station belongs to the previous profile's scene items
        inlet_items = (
            []
            if self._shares_line1
            else [self.center1, self.end1, self.line1.end2, self.line1]
        )
        return [
            self.fill,
            *inlet_items,
            self.center2,
            self.end2,
            self.line2.end2,
            self.line2,
            *self.perpendiculars,
            self.spline_a,
            self.spline_b,
        ]

    def add_to_scene(self, scene):
        """Add every item of the profile to the given scene."""
        for item in self.items:
            scene.addItem(item)
