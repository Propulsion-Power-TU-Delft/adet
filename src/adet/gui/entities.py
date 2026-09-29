import math

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsItem


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

            for dependent in self._dependents:
                dependent._on_point_moved()
            return super().itemChange(change, constrained_value)
        return super().itemChange(change, value)


class Line(QGraphicsItem):
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
        start_point: DraggablePoint,
        control_point1: DraggablePoint,
        control_point2: DraggablePoint,
        end_point: DraggablePoint,
        parent=None,
    ):
        super().__init__(parent)
        self.start_point = start_point
        self.control_point1 = control_point1
        self.control_point2 = control_point2
        self.end_point = end_point
        self.segments = 300
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)

        # Register as dependent on all control points
        self.start_point.add_dependent(self)
        self.control_point1.add_dependent(self)
        self.control_point2.add_dependent(self)
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
                    # Calculate displacement
                    old_center = self.center.pos()
                    dx = value.x() - old_center.x()
                    dy = value.y() - old_center.y()

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


class AlignedPoints(QGraphicsItem):
    """Keeps two points aligned on the same x or y coordinate.

    When one point moves, the other updates to maintain alignment on the
    specified axis ('x' or 'y').
    """

    def __init__(
        self,
        point1: DraggablePoint,
        point2: DraggablePoint,
        axis: str = 'x',
        parent=None,
    ):
        super().__init__(parent)
        self.point1 = point1
        self.point2 = point2
        self.axis = axis
        self._updating = False

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)

        self.point1.add_dependent(self)
        self.point2.add_dependent(self)

        original_p1_itemChange = self.point1.itemChange
        original_p2_itemChange = self.point2.itemChange

        def p1_constrained_itemChange(change, value):
            if (
                change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
                and not self._updating
            ):
                self._updating = True
                try:
                    if self.axis == 'x':
                        self.point2.setPos(value.x(), self.point2.pos().y())
                    else:
                        self.point2.setPos(self.point2.pos().x(), value.y())
                finally:
                    self._updating = False

            return original_p1_itemChange(change, value)

        def p2_constrained_itemChange(change, value):
            if (
                change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
                and not self._updating
            ):
                self._updating = True
                try:
                    if self.axis == 'x':
                        self.point1.setPos(value.x(), self.point1.pos().y())
                    else:
                        self.point1.setPos(self.point1.pos().x(), value.y())
                finally:
                    self._updating = False

            return original_p2_itemChange(change, value)

        self.point1.itemChange = p1_constrained_itemChange  # type: ignore
        self.point2.itemChange = p2_constrained_itemChange  # type: ignore

    def boundingRect(self):
        return QRectF()

    def paint(self, painter: QPainter, _option, _widget):
        pass

    def _on_point_moved(self):
        pass


class PerpendicularPoints(QGraphicsItem):
    """Keeps two points' connecting line perpendicular to a reference direction.

    The line formed by perp_point1 and perp_point2 stays perpendicular to the
    line formed by ref_point1 and ref_point2.
    """

    def __init__(
        self,
        ref_point1: DraggablePoint,
        ref_point2: DraggablePoint,
        perp_point1: DraggablePoint,
        perp_point2: DraggablePoint,
        parent=None,
    ):
        super().__init__(parent)
        self.ref_point1 = ref_point1
        self.ref_point2 = ref_point2
        self.perp_point1 = perp_point1
        self.perp_point2 = perp_point2
        self._updating = False

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)

        for point in [ref_point1, ref_point2, perp_point1, perp_point2]:
            point.add_dependent(self)

        original_pp1_itemChange = self.perp_point1.itemChange
        original_pp2_itemChange = self.perp_point2.itemChange

        def enforce_perpendicular():
            """Adjust perp_point2 to be perpendicular to reference direction."""
            ref1 = self.ref_point1.get_position()
            ref2 = self.ref_point2.get_position()
            pp1 = self.perp_point1.get_position()

            ref_dx = ref2.x() - ref1.x()
            ref_dy = ref2.y() - ref1.y()

            perp_dx = -ref_dy
            perp_dy = ref_dx

            perp_len = math.sqrt(perp_dx**2 + perp_dy**2)
            if perp_len == 0:
                return

            dist = math.sqrt(
                (self.perp_point2.pos().x() - pp1.x()) ** 2
                + (self.perp_point2.pos().y() - pp1.y()) ** 2
            )

            perp_dx_normalized = (perp_dx / perp_len) * dist
            perp_dy_normalized = (perp_dy / perp_len) * dist

            self._updating = True
            try:
                self.perp_point2.setPos(
                    pp1.x() + perp_dx_normalized, pp1.y() + perp_dy_normalized
                )
            finally:
                self._updating = False

        def pp1_constrained_itemChange(change, value):
            if (
                change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
                and not self._updating
            ):
                enforce_perpendicular()

            return original_pp1_itemChange(change, value)

        def pp2_constrained_itemChange(change, value):
            if (
                change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
                and not self._updating
            ):
                enforce_perpendicular()

            return original_pp2_itemChange(change, value)

        self.perp_point1.itemChange = pp1_constrained_itemChange  # type: ignore
        self.perp_point2.itemChange = pp2_constrained_itemChange  # type: ignore

    def boundingRect(self):
        return QRectF()

    def paint(self, painter: QPainter, _option, _widget):
        pass

    def _on_point_moved(self):
        pass
