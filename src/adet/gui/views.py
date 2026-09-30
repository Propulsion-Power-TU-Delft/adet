"""Views of the GUI: synced profile views, velocity triangles and the main window."""

import math
from collections.abc import Callable

from PyQt6.QtCore import (
    QEasingCurve,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QKeySequence,
    QPainter,
    QPen,
    QPolygonF,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
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
from adet.gui.labels import AngleLabel, RadiusAxis, RotationArrow
from adet.gui.row_backend import RowBackend, n0, n1
from adet.variables import VarSpec

ZOOM_STEP = 1.25
ZOOM_DURATION_MS = 150
FIT_MARGIN = 20  # scene units around the content when fitting
RADIUS_ORIGIN_Y = 550.0  # scene y of radius 0
SCENE_PER_METER = 2000.0  # the initial center at y=350 is then 0.1 m
PROFILE_START_X = 150.0  # scene x of the inlet station
PARABOLA_START_Y = 150.0  # scene y of the first point of the camber parabola
SOLVE_DELAY_MS = 30  # geometry changes within this window share one solve


def _line_points(
    x: float, radius: float, height: float, angle: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Scene center and upper end of a station line.

    ``angle`` (rad) is the tilt of the line from the vertical, positive when its
    upper end leans towards +x.
    """
    half = height / 2 * SCENE_PER_METER
    center_y = RADIUS_ORIGIN_Y - radius * SCENE_PER_METER
    return (x, center_y), (
        x + half * math.sin(angle),
        center_y - half * math.cos(angle),
    )


def _line_geometry(center: QPointF, end: QPointF) -> tuple[float, float]:
    """Height (m) and tilt from the vertical (rad) of a station line."""
    dx = end.x() - center.x()
    dy_up = center.y() - end.y()
    # Either end can be the upper one; fold the direction into the upper half plane
    if dy_up < 0 or (dy_up == 0 and dx < 0):
        dx, dy_up = -dx, -dy_up
    return 2 * math.hypot(dx, dy_up) / SCENE_PER_METER, math.atan2(dx, dy_up)


def _make_spin(
    value: float, low: float, high: float, step: float, decimals: int
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setKeyboardTracking(False)  # solve on Enter or focus loss, not per digit
    return spin


class ChangeNotifier:
    """Calls ``callback`` whenever any of the watched points moves."""

    def __init__(self, sources, callback):
        self.callback = callback
        for source in sources:
            source.add_dependent(self)

    def _on_point_moved(self):
        self.callback()


class SyncedView(QGraphicsView):
    """A fixed-scale view sharing its x range and scroll position with its group.

    Views in the same group always have the same x range and horizontal scroll
    position, and none of them rescales, so points aligned in x stay aligned on screen.
    """

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Left click + drag on empty space pans; points still grab the press first
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Default extent; the scene grows beyond it when items are dragged outside
        self._scene = scene
        self._base_rect = scene.sceneRect()
        self.group: list[SyncedView] = [self]
        self._syncing_scroll = False
        self.axis_y: float | None = None  # scene y of a dash-dot axis line, if any
        self.zoom_callback: Callable[[float], None] | None = None  # for Ctrl + scroll
        self._hbar().valueChanged.connect(self._sync_scroll)

    def drawBackground(self, painter, rect):
        """Draw the background and the dash-dot axis line across the whole view."""
        super().drawBackground(painter, rect)
        if self.axis_y is None or painter is None:
            return
        pen = QPen(QColor(150, 150, 150), 2)
        # Dash pattern in multiples of the pen width: dash, gap, dot, gap
        pen.setDashPattern([6, 5, 1, 5])
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(
            QPointF(rect.left(), self.axis_y), QPointF(rect.right(), self.axis_y)
        )

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

    def content_rect(self) -> QRectF:
        """Bounding rect of the drawn geometry, ignoring the radius axis."""
        rect = QRectF()
        for item in self._scene.items():
            top = item
            while (parent := top.parentItem()) is not None:
                top = parent
            if not isinstance(top, (RadiusAxis, RotationArrow)):
                rect = rect.united(item.sceneBoundingRect())
        return rect.adjusted(-FIT_MARGIN, -FIT_MARGIN, FIT_MARGIN, FIT_MARGIN)

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

    def wheelEvent(self, event):
        """Zoom all views with Ctrl + scroll, otherwise scroll as usual."""
        if (
            self.zoom_callback is not None
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            delta = event.angleDelta().y()
            if delta:
                self.zoom_callback(ZOOM_STEP ** (delta / 120))
            event.accept()
            return
        super().wheelEvent(event)

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


class VelocityTriangleView(QWidget):
    """Velocity triangle of one station, drawn with Qt graphics items.

    Axes follow ``plot_velocity_triangles``: V_m to the right, V_t upwards. W and V
    start at the origin and U closes the triangle.
    """

    ARROW_HEAD = 0.06  # head length as a fraction of the drawn extent
    COLORS = {
        'W': QColor(40, 110, 220),
        'U': QColor(150, 80, 200),
        'V': QColor(220, 60, 50),
    }

    def __init__(self, title: str):
        super().__init__()
        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        # The visible range is set exactly by ``fit_group``, so no scrolling
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.data_rect = QRectF()  # area the drawing needs, in scene units
        self.group: list[VelocityTriangleView] = [self]  # views sharing scale and x
        self._axis_items: list[QGraphicsItem] = []
        self._readout = QLabel('')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        layout.addWidget(self._view, 1)
        layout.addWidget(self._readout)

    def set_velocities(self, v_tan: float, v_mer: float, blade_speed: float):
        """Redraw the triangle (m/s)."""
        self._readout.setStyleSheet('')
        w_tan = v_tan - blade_speed
        self._scene.clear()
        self._axis_items = []
        # Scene y points down, so the tangential direction is flipped
        origin = QPointF(0, 0)
        v_tip = QPointF(v_mer, -v_tan)
        w_tip = QPointF(v_mer, -w_tan)
        extent = max(abs(v_mer), abs(v_tan), abs(w_tan), abs(blade_speed), 1e-6)

        self._arrow('W', origin, w_tip, extent)
        self._arrow('U', w_tip, v_tip, extent)
        self._arrow('V', origin, v_tip, extent)

        # The axes and the scale are set by ``fit_group``, once every view has its data
        ys = (0.0, -v_tan, -w_tan)
        left = min(v_mer, 0.0) - 0.15 * extent
        right = max(v_mer, 0.0) + 0.45 * extent
        self.data_rect = QRectF(
            left,
            min(ys) - 0.2 * extent,
            right - left,
            max(ys) - min(ys) + 0.4 * extent,
        )
        self._readout.setText(
            f'Vm = {v_mer:.1f}   Vt = {v_tan:.1f}   Wt = {w_tan:.1f}   '
            f'U = {blade_speed:.1f} m/s'
        )

    def set_failed(self):
        """Remove the triangle and warn that the solution failed."""
        self._scene.clear()
        self._axis_items = []
        self.data_rect = QRectF()
        self._readout.setStyleSheet('color: red; font-weight: bold;')
        self._readout.setText('FAILING')

    def _arrow(self, name: str, start: QPointF, end: QPointF, extent: float):
        color = self.COLORS[name]
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        self._scene.addLine(start.x(), start.y(), end.x(), end.y(), pen)

        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return
        head = min(self.ARROW_HEAD * extent, length / 2)
        ux, uy = dx / length, dy / length
        base = QPointF(end.x() - head * ux, end.y() - head * uy)
        normal = QPointF(-uy, ux) * (head / 3)
        self._scene.addPolygon(
            QPolygonF([end, base + normal, base - normal]), pen, QBrush(color)
        )
        label = QGraphicsSimpleTextItem(name)
        label.setBrush(QBrush(color))
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        label.setPos((start + end) / 2)
        self._scene.addItem(label)

    @staticmethod
    def _ticks(low: float, high: float, pixels: int) -> list[float]:
        """Round tick values in [low, high], about one per 70 pixels."""
        raw = (high - low) / max(pixels / 70, 1)
        magnitude = 10 ** math.floor(math.log10(raw))
        step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
        return [
            k * step for k in range(math.ceil(low / step), math.floor(high / step) + 1)
        ]

    def _draw_axes(self, rect: QRectF, scale: float):
        """Axes through the origin with ticks and labels over the visible ``rect``."""
        for item in self._axis_items:
            self._scene.removeItem(item)
        self._axis_items = []
        pen = QPen(QColor(150, 150, 150), 1)
        pen.setCosmetic(True)
        tick = 4 / scale  # tick half length of 4 pixels

        def add_label(text: str, x: float, y: float):
            label = QGraphicsSimpleTextItem(text)
            label.setBrush(QBrush(QColor('white')))
            label.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            label.setPos(x, y)
            self._scene.addItem(label)
            self._axis_items.append(label)

        def add_line(x1: float, y1: float, x2: float, y2: float):
            line = self._scene.addLine(x1, y1, x2, y2, pen)
            assert line is not None
            self._axis_items.append(line)

        viewport = self._view.viewport()
        assert viewport is not None
        add_line(rect.left(), 0, rect.right(), 0)
        add_line(0, rect.top(), 0, rect.bottom())
        for x in self._ticks(rect.left(), rect.right(), viewport.width()):
            add_line(x, -tick, x, tick)
            add_label(f'{x:g}', x - 10 / scale, 6 / scale)
        # Scene y points down, so the tangential value is the negated scene y
        for value in self._ticks(-rect.bottom(), -rect.top(), viewport.height()):
            if value == 0:
                continue
            add_line(-tick, -value, tick, -value)
            add_label(f'{value:g}', 6 / scale, -value - 8 / scale)
        add_label('Vm [m/s]', rect.right() - 60 / scale, -22 / scale)
        add_label('Vt [m/s]', 8 / scale, rect.top() + 2 / scale)

    def _apply_view(self, scale: float, center_x: float):
        """Show the scene at ``scale`` with ``center_x`` in the middle of the view."""
        viewport = self._view.viewport()
        assert viewport is not None
        width, height = viewport.width() / scale, viewport.height() / scale
        center_y = self.data_rect.center().y()
        rect = QRectF(center_x - width / 2, center_y - height / 2, width, height)
        self._scene.setSceneRect(rect)
        self._draw_axes(rect, scale)
        self._view.resetTransform()
        self._view.scale(scale, scale)
        self._view.centerOn(center_x, center_y)

    @staticmethod
    def fit_group(views: list['VelocityTriangleView']):
        """Give the views one common scale and the same x range (and x ticks)."""
        views = [view for view in views if not view.data_rect.isNull()]
        if not views:
            return
        left = min(view.data_rect.left() for view in views)
        right = max(view.data_rect.right() for view in views)
        scales = []
        for view in views:
            viewport = view._view.viewport()
            assert viewport is not None
            scales.append(
                min(
                    viewport.width() / (right - left),
                    viewport.height() / view.data_rect.height(),
                )
            )
        scale = min(scales)
        if scale <= 0:
            return
        for view in views:
            view._apply_view(scale, (left + right) / 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_group(self.group)


class MainGuiView(QWidget):
    """Meridional profile and parabola in two separate views, so lines never overlap."""

    def __init__(self, backend: RowBackend | None = None):
        super().__init__()
        self.setWindowTitle('ADeT')
        self.setGeometry(100, 100, 1200, 700)

        # First solution of the row; its geometry defines the initial drawing
        self.backend = backend if backend is not None else RowBackend()
        geometry = self._backend_geometry()

        # Meridional profile in its own scene
        profile_scene = QGraphicsScene()
        center1, end1 = _line_points(PROFILE_START_X, *geometry[0])
        center2, end2 = _line_points(
            PROFILE_START_X + geometry[2] * SCENE_PER_METER, *geometry[1]
        )
        self.profile = MeridionalProfile(
            center1=center1, end1=end1, center2=center2, end2=end2
        )
        self.profile.add_to_scene(profile_scene)
        # Radius axis left of the profile, zero at the bottom of the drawing; the
        # first center cannot go below radius 0 and starts at 0.1 m
        self.profile.center1.max_y = RADIUS_ORIGIN_Y
        self.radius_axis = RadiusAxis(
            self.profile.center1,
            axis_x=50,
            origin_y=RADIUS_ORIGIN_Y,
            scene_per_meter=SCENE_PER_METER,
            tick_step=0.05,
            min_extent=0.25,
        )
        profile_scene.addItem(self.radius_axis)
        profile_scene.setSceneRect(-100, -50, 900, 650)

        # Parabola in a second scene; the points only share x with the centers
        # through the alignment, which works across scenes
        parabola_scene = QGraphicsScene()
        center1 = self.profile.center1
        center2 = self.profile.center2
        # The camber parabola is built from the metal angles (positive = rising)
        metal0, metal1 = geometry[3]
        half_dx = (center2.pos().x() - center1.pos().x()) / 2
        control_y = PARABOLA_START_Y - half_dx * math.tan(metal0)
        end_y = control_y - half_dx * math.tan(metal1)
        start = DraggablePoint(center1.pos().x(), PARABOLA_START_Y)
        end = DraggablePoint(center2.pos().x(), end_y)
        self.start_alignment = AlignedPoints(center1, start, 'y')
        self.end_alignment = AlignedPoints(center2, end, 'y')
        control = DraggablePoint((start.pos().x() + end.pos().x()) / 2, control_y)
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
        # Rotation direction left of the parabola, in line with the radius axis
        self.rotation_arrow = RotationArrow(self.backend.get_value(n1.kin.Omega))
        self.rotation_arrow.setPos(50, 125)
        parabola_scene.addItem(self.rotation_arrow)
        parabola_scene.setSceneRect(-100, 0, 900, 250)

        self.profile_view = SyncedView(profile_scene)
        self.profile_view.axis_y = RADIUS_ORIGIN_Y  # radius 0, the axis of rotation
        self.parabola_view = SyncedView(parabola_scene)
        root = QHBoxLayout(self)
        layout = QVBoxLayout()
        root.addLayout(layout, 3)
        # Button row on the top right
        top_row = QHBoxLayout()
        self.status_label = QLabel('Initial solution converged')
        top_row.addWidget(self.status_label)
        top_row.addStretch()
        # Operating conditions, each change schedules a solve
        self.mass_flow_spin = _make_spin(
            self.backend.get_value(n0.oth.TotMassFlow), 0.0, 1e4, 0.5, 3
        )
        self.omega_spin = _make_spin(
            self.backend.get_value(n1.kin.Omega), -1e5, 1e5, 10.0, 1
        )
        top_row.addWidget(QLabel('Mass flow [kg/s]'))
        top_row.addWidget(self.mass_flow_spin)
        top_row.addWidget(QLabel('Omega [rad/s]'))
        top_row.addWidget(self.omega_spin)
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
            view.zoom_callback = self.zoom_views

        # Inlet and outlet velocity triangles in a column on the right
        self.triangle_views = (
            VelocityTriangleView('Inlet velocity triangle'),
            VelocityTriangleView('Outlet velocity triangle'),
        )
        side = QVBoxLayout()
        for triangle_view in self.triangle_views:
            triangle_view.group = list(self.triangle_views)
            side.addWidget(triangle_view)
        root.addLayout(side, 1)
        self._update_triangles()

        self.mass_flow_spin.valueChanged.connect(self._schedule_solve)
        self.omega_spin.valueChanged.connect(self._schedule_solve)
        self.omega_spin.valueChanged.connect(self.rotation_arrow.set_omega)

        # Any geometry change schedules one Newton solve; changes arriving while it
        # is pending are merged, so dragging does not queue up solves
        self._solve_timer = QTimer(self)
        self._solve_timer.setSingleShot(True)
        self._solve_timer.setInterval(SOLVE_DELAY_MS)
        self._solve_timer.timeout.connect(self.update_solution)
        self._notifier = ChangeNotifier(
            (
                self.profile.center1,
                self.profile.end1,
                self.profile.center2,
                self.profile.end2,
                start,
                control,
                end,
            ),
            self._schedule_solve,
        )

        # Animation runs on the log of the zoom so steps compose smoothly
        self._zoom_applied = 0.0
        self._zoom_target = 0.0
        self._zoom_anim = QVariantAnimation(self)
        self._zoom_anim.setDuration(ZOOM_DURATION_MS)
        self._zoom_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._zoom_anim.valueChanged.connect(self._on_zoom_step)

        # Ctrl + / Ctrl - zoom both views together
        for keys, factor in (
            (('Ctrl++', 'Ctrl+='), ZOOM_STEP),
            (('Ctrl+-',), 1 / ZOOM_STEP),
        ):
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(lambda f=factor: self.zoom_views(f))
        # Ctrl 0 fits both views, like the button
        QShortcut(QKeySequence('Ctrl+0'), self).activated.connect(self.fit_views)

    def _backend_geometry(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        float,
        tuple[float, float],
    ]:
        """Read the geometry of the first solution.

        Returns (radius, height, meridional angle) of the inlet and of the outlet,
        the axial chord, and the inlet and outlet metal angles.
        """
        get = self.backend.get_value
        inlet = (
            get(n0.geo.Rmid),
            get(n0.geo.Height),
            get(n0.geo.MeridionalAngle),
        )
        outlet = (
            get(n1.geo.Rmid),
            get(n1.geo.Height),
            get(n1.geo.MeridionalAngle),
        )
        return (
            inlet,
            outlet,
            get(n1.geo.ChordAx),
            (get(n0.geo.MetalAngle), get(n1.geo.MetalAngle)),
        )

    def drawn_geometry(self) -> dict[VarSpec, float]:
        """Geometry currently drawn, as boundary conditions of the row (m, rad)."""
        profile = self.profile
        height0, mer_angle0 = _line_geometry(
            profile.center1.get_position(), profile.end1.get_position()
        )
        height1, mer_angle1 = _line_geometry(
            profile.center2.get_position(), profile.end2.get_position()
        )
        return {
            n0.geo.Rmid: self.radius_axis.radius(),
            n0.geo.Height: height0,
            n0.geo.MeridionalAngle: mer_angle0,
            n1.geo.Rmid: (RADIUS_ORIGIN_Y - profile.center2.get_position().y())
            / SCENE_PER_METER,
            n1.geo.Height: height1,
            n1.geo.MeridionalAngle: mer_angle1,
            n1.geo.ChordAx: (
                profile.center2.get_position().x() - profile.center1.get_position().x()
            )
            / SCENE_PER_METER,
            # Scene y points down, metal angles are positive when rising
            n0.geo.MetalAngle: -math.radians(self.parabola.inlet_angle),
            n1.geo.MetalAngle: -math.radians(self.parabola.outlet_angle),
        }

    def operating_conditions(self) -> dict[VarSpec, float]:
        """Mass flow (kg/s) and rotational speed (rad/s) from the input fields."""
        return {
            n0.oth.TotMassFlow: self.mass_flow_spin.value(),
            n1.kin.Omega: self.omega_spin.value(),
        }

    def _schedule_solve(self):
        if not self._solve_timer.isActive():
            self._solve_timer.start()

    def update_solution(self):
        """Pass the drawn geometry to ``kn`` and re-solve the row with Newton."""
        self.backend.set_geometry(self.drawn_geometry() | self.operating_conditions())
        converged = self.backend.solve()
        if converged:
            self.status_label.setText('Newton converged')
        else:
            self.status_label.setText('Newton failed')
        self._update_triangles(converged)

    def _update_triangles(self, converged: bool = True):
        """Show the velocity triangles of the current solution."""
        if not converged:
            for triangle_view in self.triangle_views:
                triangle_view.set_failed()
            return
        get = self.backend.get_value
        for triangle_view, node in zip(self.triangle_views, (n0, n1)):
            triangle_view.set_velocities(
                get(node.kin.V_tan), get(node.kin.V_mer), get(node.kin.BladeSpeed)
            )
        VelocityTriangleView.fit_group(list(self.triangle_views))

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
        """Fit the profile and the parabola (not the axis) using one common scale."""
        self.profile_view._update_extent()
        rects = [view.content_rect() for view in self.group]
        # The x range is shared, so it must hold the content of every view
        left = min(rect.left() for rect in rects)
        right = max(rect.right() for rect in rects)
        width = right - left
        scales = []
        for view, rect in zip(self.group, rects):
            viewport = view.viewport()
            assert viewport is not None
            scales.append(
                min(viewport.width() / width, viewport.height() / rect.height())
            )
        scale = min(scales)
        center_x = (left + right) / 2
        for view, rect in zip(self.group, rects):
            view.resetTransform()
            view.scale(scale, scale)
            view.centerOn(center_x, rect.center().y())
