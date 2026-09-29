import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QGraphicsItem


CONTROL_MARKER_RADIUS = 4.0
ANGLE_SNAP_DEG = 10.0


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
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self._dependents = []

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
    ):
        super().__init__(parent)
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

    def _get_bezier_point(self, t: float) -> QPointF:
        """Calculate a point on the quadratic Bezier curve at parameter t (0-1)."""
        p0 = self.start_point.get_position()
        p1 = self.control_point.get_position()
        p2 = self.end_point.get_position()

        x = (1 - t) ** 2 * p0.x() + 2 * (1 - t) * t * p1.x() + t**2 * p2.x()
        y = (1 - t) ** 2 * p0.y() + 2 * (1 - t) * t * p1.y() + t**2 * p2.y()
        return QPointF(x, y)

    def boundingRect(self):
        points = [
            self._get_bezier_point(i / self.segments) for i in range(self.segments + 1)
        ]
        if not points:
            return QRectF()
        min_x = min(p.x() for p in points) - 2
        min_y = min(p.y() for p in points) - 2
        max_x = max(p.x() for p in points) + 2
        max_y = max(p.y() for p in points) + 2
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter: QPainter, _option, _widget):
        painter.setPen(QPen(QColor(100, 200, 100), 2))
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
    ):
        super().__init__(parent)
        self.show_control_polygon = show_control_polygon
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

        painter.setPen(QPen(QColor(200, 150, 100), 2))
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
    ):
        super().__init__(parent)
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
        if not (QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier):
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
        painter.setPen(QPen(QColor(100, 150, 200), 2))
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


class MeridionalProfile:
    """A meridional profile made of two symmetric lines joined by two splines.

    Each symmetric line has a draggable center and two mirrored endpoints. Each
    endpoint has a perpendicular constraint whose calculated point acts as a
    control point of a cubic Bezier spline, so the splines leave both lines
    at a right angle. The first spline joins the ``end1`` points of the lines
    and the second joins the ``end2`` points.
    """

    def __init__(
        self,
        center1: tuple[float, float] = (150, 600),
        end1: tuple[float, float] = (150, 400),
        center2: tuple[float, float] = (550, 400),
        end2: tuple[float, float] = (550, 200),
        control_fraction: float = 1 / 3,
        center1_axis_constraint: str | None = 'y',
    ):
        # First line
        self.center1 = DraggablePoint(*center1, axis_constraint=center1_axis_constraint)
        self.end1 = DraggablePoint(*end1)
        self.line1 = SymmetricLine(self.center1, self.end1)

        # Second line
        # The second center cannot go left of the first center
        self.center2 = DraggablePoint(*center2, min_x=self.center1.pos().x())
        self.end2 = DraggablePoint(*end2)
        self.line2 = SymmetricLine(self.center2, self.end2)

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
        )
        self.perp1_a.add_dependent(self.spline_a)
        self.perp2_a.add_dependent(self.spline_a)

        self.spline_b = CubicBezierLine(
            self.line1.end2,
            self.perp1_b.point3,
            self.perp2_b.point3,
            self.line2.end2,
            show_control_polygon=True,
        )
        self.perp1_b.add_dependent(self.spline_b)
        self.perp2_b.add_dependent(self.spline_b)

        # The constraints still drive the control points when hidden
        for perp in self.perpendiculars:
            perp.setVisible(False)

    @property
    def perpendiculars(self) -> tuple[PerpendicularPoints, ...]:
        return self.perp1_a, self.perp1_b, self.perp2_a, self.perp2_b

    @property
    def items(self) -> list[QGraphicsItem]:
        """All graphics items of the profile, in the order they should be added."""
        return [
            self.center1,
            self.end1,
            self.line1.end2,
            self.line1,
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
