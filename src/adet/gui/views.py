"""Views of the GUI: synced profile views, velocity triangles and the main window."""

import logging
import math
from collections.abc import Callable

from PyQt6.QtCore import (
    QEasingCurve,
    QEvent,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRegion,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSplitter,
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
from adet.gui.row_backend import RowBackend, n0
from adet.variables import VarSpec

logger = logging.getLogger(__name__)

# *** Animations
ZOOM_STEP = 1.25
ZOOM_DURATION_MS = 150
FIT_DURATION_MS = 100  # glide of the views to the fit when a row is added

FIT_MARGIN = 20  # scene units around the content when fitting
RADIUS_ORIGIN_Y = 550.0  # scene y of radius 0
SCENE_PER_METER = 2000.0  # the initial center at y=350 is then 0.1 m
PROFILE_START_X = 150.0  # scene x of the inlet station
PARABOLA_START_Y = 150.0  # scene y of the first point of the camber parabola
AXIAL_GAP = 20.0  # visual gap between blade rows along the wall, scene units
TRIANGLE_COLUMN_WIDTH = 350  # window growth per added row, pixels
MAX_VISIBLE_ROWS = 2  # rows whose triangles grow the window; more rows scroll
ROW_GAP = 16  # spacing between the widgets of neighbouring blade rows, pixels
SOLVE_DELAY_MS = 30  # geometry changes within this window share one solve
BACKGROUND_COLOR = QColor(15, 15, 15)  # background of every view
CASING_COLOR = QColor(170, 170, 170)
ROTOR_COLOR = QColor(70, 170, 255)
MOVING_SHAFT_COLORS = (  # colors given in turn to the moving shafts
    ROTOR_COLOR,
    QColor(255, 170, 60),
    QColor(110, 210, 120),
    QColor(230, 100, 200),
    QColor(240, 220, 80),
)
SHAFT_PANEL_WIDTH = 240  # pixels
SHAFT_CIRCLE_SIZE = 18  # pixels
VIEW_CORNER_RADIUS = 16.0  # corner rounding of every view, pixels
SPLITTER_HANDLE_WIDTH = 10  # gap between views, wide enough to grab
PROFILE_VIEW_STRETCH = 3  # startup height share of the meridional profile view
PARABOLA_VIEW_STRETCH = 2  # startup height share of the camber line view


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


def _circle_pixmap(color: QColor, size: int = SHAFT_CIRCLE_SIZE) -> QPixmap:
    """A filled circle of ``color`` on a transparent background."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(color))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return pixmap


class ShaftView(QWidget):
    """One shaft: a circle in its color, its name and its rotational speed.

    A removable shaft has a button that fires ``remove_requested``.
    """

    remove_requested = pyqtSignal()

    def __init__(
        self,
        name: str,
        color: QColor,
        omega: float,
        editable: bool = True,
        removable: bool = False,
    ):
        super().__init__()
        self.name = name
        self.color = color
        circle = QLabel()
        circle.setPixmap(_circle_pixmap(color))
        title = QLabel(name)
        title.setStyleSheet('font-weight: bold;')
        self.omega_spin = _make_spin(omega, -1e5, 1e5, 10.0, 1)
        self.omega_spin.setEnabled(editable)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title_row = QHBoxLayout()
        title_row.addWidget(circle)
        title_row.addWidget(title, 1)
        if removable:
            remove_button = QPushButton('×')
            remove_button.setFixedSize(SHAFT_CIRCLE_SIZE + 6, SHAFT_CIRCLE_SIZE + 6)
            remove_button.setToolTip('Remove shaft')
            remove_button.clicked.connect(self.remove_requested)
            title_row.addWidget(remove_button)
        layout.addLayout(title_row)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel('omega [rad/s]'))
        speed_row.addWidget(self.omega_spin, 1)
        layout.addLayout(speed_row)

    @property
    def omega(self) -> float:
        return self.omega_spin.value()


class ShaftPanel(QWidget):
    """Column of shafts on the left of the window.

    ``changed`` fires on any speed edit, ``shaft_added`` with every new shaft and
    ``shaft_removed`` with the list index and the shaft removed.
    """

    changed = pyqtSignal()
    shaft_added = pyqtSignal(ShaftView)
    shaft_removed = pyqtSignal(int, ShaftView)

    def __init__(self):
        super().__init__()
        self.shafts: list[ShaftView] = []
        self._next_number = 2  # the first moving shaft is added without a number
        self.setMinimumWidth(SHAFT_PANEL_WIDTH)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(ROW_GAP)
        title = QLabel('Shafts')
        title.setStyleSheet('font-weight: bold; font-size: 14px;')
        self._layout.addWidget(title)
        add_button = QPushButton('+ Add moving shaft')
        add_button.clicked.connect(self.add_moving_shaft)
        self._layout.addWidget(add_button)
        self._layout.addStretch()

    def add_shaft(
        self,
        name: str,
        color: QColor,
        omega: float,
        editable: bool = True,
        removable: bool = False,
    ) -> ShaftView:
        shaft = ShaftView(name, color, omega, editable, removable)
        shaft.omega_spin.valueChanged.connect(self.changed)
        shaft.remove_requested.connect(lambda: self.remove_shaft(shaft))
        # Above the add button and the stretch
        self._layout.insertWidget(len(self.shafts) + 1, shaft)
        self.shafts.append(shaft)
        self.shaft_added.emit(shaft)
        return shaft

    def add_moving_shaft(self) -> ShaftView:
        """Add a removable shaft turning as fast as the last one.

        It takes the first color of the palette that no shaft uses, or cycles
        through the palette when all are taken.
        """
        used = [shaft.color for shaft in self.shafts]
        free = [color for color in MOVING_SHAFT_COLORS if color not in used]
        color = (
            free[0]
            if free
            else MOVING_SHAFT_COLORS[len(used) % len(MOVING_SHAFT_COLORS)]
        )
        name = f'Rotating shaft {self._next_number}'
        self._next_number += 1
        return self.add_shaft(name, color, self.shafts[-1].omega, removable=True)

    def remove_shaft(self, shaft: ShaftView):
        """Take a shaft out of the panel and tell the listeners."""
        index = self.shafts.index(shaft)
        self.shafts.pop(index)
        self._layout.removeWidget(shaft)
        shaft.hide()  # until the deferred delete happens
        shaft.deleteLater()
        self.shaft_removed.emit(index, shaft)


class ChangeNotifier:
    """Calls ``callback`` whenever any of the watched points moves."""

    def __init__(self, sources, callback):
        self.callback = callback
        for source in sources:
            source.add_dependent(self)

    def _on_point_moved(self):
        self.callback()


class RowButton(QGraphicsItem):
    """A round ``+`` or ``-`` button with a constant pixel size in a scene.

    ``offset`` (pixels) is the top left corner of the button relative to its position.
    """

    SIZE = 36  # pixels
    DISABLED_COLOR = QColor(90, 90, 90)

    def __init__(
        self, plus: bool, color: QColor, callback: Callable[[], None], offset: QPointF
    ):
        super().__init__()
        self.plus = plus
        self.color = color
        self.callback = callback
        self.offset = offset
        self.enabled = True
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(10)

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def boundingRect(self):
        return QRectF(self.offset.x(), self.offset.y(), self.SIZE, self.SIZE)

    def paint(self, painter: QPainter, _option, _widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.color if self.enabled else self.DISABLED_COLOR
        painter.setPen(QPen(color, 3))
        painter.setBrush(QBrush(BACKGROUND_COLOR))
        rect = self.boundingRect().adjusted(2, 2, -2, -2)
        # painter.drawEllipse(rect)
        center, arm = rect.center(), self.SIZE / 4
        painter.drawLine(center - QPointF(arm, 0), center + QPointF(arm, 0))
        if self.plus:
            painter.drawLine(center - QPointF(0, arm), center + QPointF(0, arm))

    def mousePressEvent(self, event):
        # Accepting the press keeps the view from starting a pan
        event.accept()
        if self.enabled:
            self.callback()


class RowControls:
    """Big ``+`` and ``-`` buttons that follow the outlet of the last row."""

    OFFSET_X = 24  # pixels to the right of the outlet station
    GAP = 4  # pixels between the buttons and the mean line

    def __init__(
        self,
        scene: QGraphicsScene,
        on_add: Callable[[], None],
        on_delete: Callable[[], None],
    ):
        size = RowButton.SIZE
        self.add_button = RowButton(
            True, QColor(90, 200, 110), on_add, QPointF(self.OFFSET_X, -size - self.GAP)
        )
        self.delete_button = RowButton(
            False, QColor(225, 80, 70), on_delete, QPointF(self.OFFSET_X, self.GAP)
        )
        self._anchor: DraggablePoint | None = None
        self._edge_end: DraggablePoint | None = None
        for button in (self.add_button, self.delete_button):
            scene.addItem(button)

    def attach(
        self, anchor: DraggablePoint, edge_end: DraggablePoint, can_delete: bool
    ):
        """Follow ``anchor``, the outlet center of the last row, and tilt with its edge.

        ``edge_end`` is an end of the outlet edge through ``anchor``.
        """
        for point in (self._anchor, self._edge_end):
            if point is not None:
                point.remove_dependent(self)
        self._anchor, self._edge_end = anchor, edge_end
        anchor.add_dependent(self)
        edge_end.add_dependent(self)
        self.delete_button.set_enabled(can_delete)
        self._on_point_moved()

    def _on_point_moved(self):
        if self._anchor is None or self._edge_end is None:
            return
        _height, tilt = _line_geometry(self._anchor.pos(), self._edge_end.pos())
        for button in (self.add_button, self.delete_button):
            button.setPos(self._anchor.pos())
            button.setRotation(math.degrees(tilt))


class RoundedGraphicsView(QGraphicsView):
    """A graphics view whose viewport has rounded corners of ``VIEW_CORNER_RADIUS``."""

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setFrameShape(QFrame.Shape.NoFrame)  # a square border would stick out

    def viewportEvent(self, event):
        if event is not None and event.type() == QEvent.Type.Resize:
            self._round_viewport()
        return super().viewportEvent(event)

    def _round_viewport(self):
        viewport = self.viewport()
        assert viewport is not None
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(viewport.rect()), VIEW_CORNER_RADIUS, VIEW_CORNER_RADIUS
        )
        viewport.setMask(QRegion(path.toFillPolygon().toPolygon()))


class SyncedView(RoundedGraphicsView):
    """A fixed-scale view sharing its x range and scroll position with its group.

    Views in the same group always have the same x range and horizontal scroll
    position, and none of them rescales, so points aligned in x stay aligned on screen.
    """

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        scene.setBackgroundBrush(QBrush(BACKGROUND_COLOR))
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
        self._scene.setBackgroundBrush(QBrush(BACKGROUND_COLOR))
        self._view = RoundedGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        # The visible range is set exactly by ``fit_group``, so no scrolling
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.data_rect = QRectF()  # area the drawing needs, in scene units
        self.group: list[VelocityTriangleView] = [self]  # views sharing scale and x
        self._axis_items: list[QGraphicsItem] = []
        # Velocities on screen (v_tan, v_mer, U) and the animation towards new ones
        self._shown: tuple[float, float, float] | None = None
        self._anim_start = (0.0, 0.0, 0.0)
        self._anim_target = (0.0, 0.0, 0.0)
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(100)
        self._anim.setEasingCurve(QEasingCurve.Type.OutBounce)
        self._anim.valueChanged.connect(self._animate_step)
        self._readout = QLabel('')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet('font-weight: bold;')
        layout.addWidget(self._title_label)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._readout)

    def set_title_color(self, color: QColor):
        self._title_label.setStyleSheet(f'font-weight: bold; color: {color.name()};')

    def set_velocities(self, v_tan: float, v_mer: float, blade_speed: float):
        """Animate the triangle from what is shown now to the new velocities (m/s)."""
        target = (v_tan, v_mer, blade_speed)
        self._anim.stop()
        start = self._shown
        if start is None:
            self._draw(*target)
            return
        self._anim_start, self._anim_target = start, target
        self._anim.start()

    def _animate_step(self, progress: float):
        """Draw the triangle a fraction ``progress`` of the way to the target."""
        current = tuple(
            a + (b - a) * progress for a, b in zip(self._anim_start, self._anim_target)
        )
        self._draw(*current)
        self.fit_group(self.group)

    def _draw(self, v_tan: float, v_mer: float, blade_speed: float):
        """Redraw the triangle (m/s)."""
        self._shown = (v_tan, v_mer, blade_speed)
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
        self._anim.stop()
        self._shown = None
        self._scene.clear()
        self._axis_items = []
        self.data_rect = QRectF()
        self._readout.setStyleSheet('color: red; font-weight: bold;')
        self._readout.setText('FAILING')

    def _arrow(self, name: str, start: QPointF, end: QPointF, extent: float):
        color = self.COLORS[name]
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
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

        grid_pen = QPen(QColor(70, 70, 70), 1)
        grid_pen.setCosmetic(True)

        def add_line(x1: float, y1: float, x2: float, y2: float, grid: bool = False):
            line = self._scene.addLine(x1, y1, x2, y2, grid_pen if grid else pen)
            assert line is not None
            if grid:
                line.setZValue(-1)  # behind the arrows
            self._axis_items.append(line)

        viewport = self._view.viewport()
        assert viewport is not None
        add_line(rect.left(), 0, rect.right(), 0)
        add_line(0, rect.top(), 0, rect.bottom())
        for x in self._ticks(rect.left(), rect.right(), viewport.width()):
            add_line(x, rect.top(), x, rect.bottom(), grid=True)
            add_line(x, -tick, x, tick)
            add_label(f'{x:g}', x - 10 / scale, 6 / scale)
        # Scene y points down, so the tangential value is the negated scene y
        for value in self._ticks(-rect.bottom(), -rect.top(), viewport.height()):
            if value == 0:
                continue
            add_line(rect.left(), -value, rect.right(), -value, grid=True)
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


class BladeRowView:
    """One blade row: meridional profile, camber line, shaft choice and triangles.

    The row is drawn into the shared ``profile_scene`` and ``parabola_scene``. Its
    inlet station is the outlet station of ``previous`` (the same points), so a chain
    of rows stays connected while any of them is edited. The row takes the color and
    the rotational speed of its shaft.
    """

    def __init__(
        self,
        backend: RowBackend,
        index: int,
        previous: 'BladeRowView | None',
        profile_scene: QGraphicsScene,
        parabola_scene: QGraphicsScene,
        shafts: list[ShaftView],
        shaft: ShaftView,
    ):
        self.backend = backend
        self.index = index
        self.previous = previous
        self.shafts = shafts
        self.shaft = shaft
        self.profile_scene = profile_scene
        self.parabola_scene = parabola_scene
        self.inlet, self.outlet = backend.row_nodes(index)
        get = backend.get_value

        # Meridional profile in the profile scene
        chord = get(self.outlet.geo.ChordAx) * SCENE_PER_METER
        # A following row is drawn after a visual gap, which the solution ignores
        self.gap = 0.0 if previous is None else AXIAL_GAP
        start_x = (
            PROFILE_START_X
            if previous is None
            else previous.profile.center2.pos().x() + self.gap
        )
        center2, end2 = _line_points(
            start_x + chord,
            get(self.outlet.geo.Rmid),
            get(self.outlet.geo.Height),
            get(self.outlet.geo.MeridionalAngle),
        )
        if previous is None:
            center1, end1 = _line_points(
                start_x,
                get(self.inlet.geo.Rmid),
                get(self.inlet.geo.Height),
                get(self.inlet.geo.MeridionalAngle),
            )
            self.profile = MeridionalProfile(
                center1=center1, end1=end1, center2=center2, end2=end2
            )
        else:
            self.profile = MeridionalProfile(
                center2=center2, end2=end2, previous=previous.profile, gap=self.gap
            )
        self.profile.add_to_scene(profile_scene)

        # Only the first row has a radius axis: its first center cannot go below
        # radius 0 and starts at 0.1 m
        self.radius_axis: RadiusAxis | None = None
        if previous is None:
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

        # Parabola in the parabola scene; the points only share x with the centers
        # through the alignment, which works across scenes
        center1 = self.profile.center1
        center2 = self.profile.center2
        # The camber parabola is built from the metal angles (positive = rising)
        metal0 = get(self.inlet.geo.MetalAngle)
        metal1 = get(self.outlet.geo.MetalAngle)
        half_dx = (center2.pos().x() - center1.pos().x()) / 2
        # A following row starts at the height where the previous camber line ends
        start_y = (
            PARABOLA_START_Y
            if previous is None
            else previous.camber_points[2].pos().y()
        )
        control_y = start_y - half_dx * math.tan(metal0)
        end_y = control_y - half_dx * math.tan(metal1)
        start = DraggablePoint(center1.pos().x(), start_y)
        end = DraggablePoint(center2.pos().x(), end_y)
        self.start_alignment = AlignedPoints(center1, start, 'y')
        self.end_alignment = AlignedPoints(center2, end, 'y')
        control = DraggablePoint((start.pos().x() + end.pos().x()) / 2, control_y)
        self.parabola = ParabolicLine(start, control, end, show_control_polygon=True)
        for item in (start, end, control, self.parabola):
            parabola_scene.addItem(item)
        self.camber_points = (start, control, end)
        self.camber_alignment: AlignedPoints | None = None
        if previous is not None:
            # The leading edge stays at the height of the previous trailing edge in
            # real time; the inlet angle follows the solution (``sync_inlet_angle``)
            self.camber_alignment = AlignedPoints(previous.camber_points[2], start, 'x')
            # The inlet angle is a result, so the control point is not draggable either
            for point in (start, control):
                point.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

        # Angle counters next to the first and last point of the parabola
        self.inlet_label = AngleLabel(
            start,
            lambda: self.parabola.inlet_angle,
            QPointF(-30, 10),
            self.camber_points,
        )
        self.outlet_label = AngleLabel(
            end,
            lambda: self.parabola.outlet_angle,
            QPointF(10, 10),
            self.camber_points,
        )
        # The leading edge of a following row sits on the trailing edge of the previous
        # one, so only the trailing edge angle of the previous row is shown
        self.inlet_label.setVisible(previous is None)
        parabola_scene.addItem(self.inlet_label)
        parabola_scene.addItem(self.outlet_label)
        # Rotation direction beneath the mid point of the parabola
        self.rotation_arrow = RotationArrow(shaft.omega, self.parabola, 215)
        parabola_scene.addItem(self.rotation_arrow)

        # Choice of the shaft the row belongs to
        self.shaft_combo = QComboBox()
        for option in shafts:
            self.shaft_combo.addItem(QIcon(_circle_pixmap(option.color)), option.name)
        self.shaft_combo.setCurrentIndex(shafts.index(shaft))
        self.shaft_combo.currentIndexChanged.connect(self._on_shaft_selected)

        # Inlet and outlet velocity triangles
        self.triangle_views = (
            VelocityTriangleView(f'Row {index + 1} inlet'),
            VelocityTriangleView(f'Row {index + 1} outlet'),
        )
        self._apply_shaft_color()

    def add_shaft_option(self, shaft: ShaftView):
        """Offer a shaft added after the row was created."""
        self.shaft_combo.addItem(QIcon(_circle_pixmap(shaft.color)), shaft.name)

    def remove_shaft_option(self, index: int, removed: ShaftView, fallback: ShaftView):
        """Drop the option at ``index``; a row on the removed
        shaft moves to ``fallback``.

        ``shafts`` no longer holds the removed shaft when this is called.
        """
        if self.shaft is removed:
            self.shaft = fallback
        # Removing the current item would otherwise select another shaft by index
        self.shaft_combo.blockSignals(True)
        self.shaft_combo.removeItem(index)
        self.shaft_combo.setCurrentIndex(self.shafts.index(self.shaft))
        self.shaft_combo.blockSignals(False)
        self._apply_shaft_color()
        self.update_rotation()

    def _on_shaft_selected(self, index: int):
        self.shaft = self.shafts[index]
        self._apply_shaft_color()
        self.update_rotation()

    def _apply_shaft_color(self):
        """Draw the profile, the camber line and
        the titles in the color of the shaft."""
        color = self.shaft.color
        self.profile.set_color(color)
        self.parabola.color = color
        self.parabola.update()
        for triangle_view in self.triangle_views:
            triangle_view.set_title_color(color)

    def update_rotation(self):
        """Point the rotation arrow the way the shaft turns."""
        self.rotation_arrow.set_omega(self.shaft.omega)

    def remove(self, notifier: ChangeNotifier):
        """Take the row out of both scenes and unhook it from the previous row.

        The outlet station of the previous row is the inlet of this one, so its
        points stay, but they must stop notifying what belonged to this row.
        """
        assert self.previous is not None, 'The first row cannot be removed'
        owners: list = [
            notifier,
            self.start_alignment,
            self.end_alignment,
            *self.profile.perpendiculars,
            self.profile.spline_a,
            self.profile.spline_b,
        ]
        if self.profile.gap_follower is not None:
            owners.append(self.profile.gap_follower)
        shared = self.previous.profile
        for point in (shared.center2, shared.line2.end1, shared.line2.end2):
            for owner in owners:
                point.remove_dependent(owner)
        if self.camber_alignment is not None:
            self.previous.camber_points[2].remove_dependent(self.camber_alignment)
        for item in (
            *self.profile.items,
            *self.camber_points,
            self.parabola,
            self.inlet_label,
            self.outlet_label,
            self.rotation_arrow,
        ):
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)

    @property
    def points(self) -> tuple[DraggablePoint, ...]:
        """Every point whose movement changes the geometry of the row."""
        profile = self.profile
        return (
            profile.center1,
            profile.end1,
            profile.center2,
            profile.end2,
            *self.camber_points,
        )

    def drawn_geometry(self) -> dict[VarSpec, float]:
        """Geometry currently drawn, as boundary conditions of the row (m, rad).

        The inlet geometry of a row that follows another one comes from the link.
        """
        profile = self.profile
        center1 = profile.center1.get_position()
        center2 = profile.center2.get_position()
        height1, mer_angle1 = _line_geometry(center2, profile.end2.get_position())
        geometry = {
            self.outlet.geo.Rmid: (RADIUS_ORIGIN_Y - center2.y()) / SCENE_PER_METER,
            self.outlet.geo.Height: height1,
            self.outlet.geo.MeridionalAngle: mer_angle1,
            # The gap lies before center1, so it is not part of the chord
            self.outlet.geo.ChordAx: (center2.x() - center1.x()) / SCENE_PER_METER,
            # Scene y points down, metal angles are positive when rising
            self.outlet.geo.MetalAngle: -math.radians(self.parabola.outlet_angle),
        }
        if self.previous is None:
            # A following row has zero incidence, so its inlet angle is a result
            geometry[self.inlet.geo.MetalAngle] = -math.radians(
                self.parabola.inlet_angle
            )
        if self.radius_axis is not None:
            height0, mer_angle0 = _line_geometry(center1, profile.end1.get_position())
            geometry |= {
                self.inlet.geo.Rmid: self.radius_axis.radius(),
                self.inlet.geo.Height: height0,
                self.inlet.geo.MeridionalAngle: mer_angle0,
            }
        return geometry

    def sync_inlet_angle(self):
        """Tilt the leading edge of a following row to the solved inlet metal angle.

        The metal angle equals the flow angle (zero incidence). The leading edge is
        pinned in y to the previous row, so only the control point moves; the end
        point stays where it was dragged.
        """
        if self.previous is None:
            return
        start, control, _end = self.camber_points
        half_dx = control.pos().x() - start.pos().x()  # control sits at the midpoint
        metal0 = self.backend.get_value(self.inlet.geo.MetalAngle)
        shift = start.pos().y() - half_dx * math.tan(metal0) - control.pos().y()
        control.setPos(control.pos().x(), control.pos().y() + shift)

    def operating_conditions(self) -> dict[VarSpec, float]:
        """Rotational speed (rad/s) of the shaft of the row."""
        return {self.outlet.kin.Omega: self.shaft.omega}

    def update_triangles(self, converged: bool = True):
        """Show the velocity triangles of the current solution."""
        if not converged:
            for triangle_view in self.triangle_views:
                triangle_view.set_failed()
            return
        get = self.backend.get_value
        for triangle_view, node in zip(self.triangle_views, (self.inlet, self.outlet)):
            triangle_view.set_velocities(
                get(node.kin.V_tan), get(node.kin.V_mer), get(node.kin.BladeSpeed)
            )


class MainGuiView(QWidget):
    """Chain of blade rows in two shared views, so lines never overlap.

    The meridional profiles of all rows are in one view and the camber lines in
    another. Each row has its own pair of velocity triangles. The ``+`` button adds
    a row after the last one.
    """

    def __init__(self, backend: RowBackend | None = None):
        super().__init__()
        self.setWindowTitle('ADeT')
        self.setGeometry(100, 100, 1300 + SHAFT_PANEL_WIDTH, 1000)

        # First solution of the row; its geometry defines the initial drawing
        self.backend = backend if backend is not None else RowBackend()
        self.rows: list[BladeRowView] = []

        self.profile_scene = QGraphicsScene()
        self.profile_scene.setSceneRect(-100, -50, 900, 650)
        self.parabola_scene = QGraphicsScene()
        self.parabola_scene.setSceneRect(-100, 0, 900, 300)
        self.profile_view = SyncedView(self.profile_scene)
        self.profile_view.axis_y = RADIUS_ORIGIN_Y  # radius 0, the axis of rotation
        self.parabola_view = SyncedView(self.parabola_scene)
        self.row_controls = RowControls(
            self.profile_scene, self.add_row, self.delete_row
        )
        # Triangle column of each row, to remove with the row
        self._row_ui: list[QVBoxLayout] = []

        # Shafts on the left; the casing is stationary and the rotating shaft starts
        # at the speed of the first row
        self.shaft_panel = ShaftPanel()
        self.casing_shaft = self.shaft_panel.add_shaft(
            'Casing (stationary)', CASING_COLOR, 0.0, False
        )
        self.rotor_shaft = self.shaft_panel.add_shaft(
            'Rotating shaft',
            ROTOR_COLOR,
            self.backend.get_value(RowBackend.row_nodes(0)[1].kin.Omega),
        )
        self.shaft_panel.changed.connect(self._on_shaft_speed_changed)
        self.shaft_panel.shaft_added.connect(self._on_shaft_added)
        self.shaft_panel.shaft_removed.connect(self._on_shaft_removed)

        # Panels are separated by draggable splitter handles
        self._root = QSplitter(Qt.Orientation.Horizontal)
        root_layout = QHBoxLayout(self)
        root_layout.addWidget(self._root)
        self._root.addWidget(self.shaft_panel)
        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        self._root.addWidget(center)
        self._root.setStretchFactor(1, 3)
        self._root.setChildrenCollapsible(False)
        self._root.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        # Button row on the top right
        top_row = QHBoxLayout()
        self.status_label = QLabel('Initial solution converged')
        top_row.addWidget(self.status_label)
        top_row.addStretch()
        # Operating conditions, each change schedules a solve
        self.mass_flow_spin = _make_spin(
            self.backend.get_value(n0.oth.TotMassFlow), 0.0, 1e4, 0.5, 3
        )
        self.mass_flow_spin.valueChanged.connect(self._schedule_solve)
        top_row.addWidget(QLabel('Mass flow [kg/s]'))
        top_row.addWidget(self.mass_flow_spin)
        fit_button = QPushButton('Fit both views')
        fit_button.clicked.connect(lambda _checked=False: self.fit_views())
        top_row.addWidget(fit_button)
        layout.addLayout(top_row)
        views_splitter = QSplitter(Qt.Orientation.Vertical)
        views_splitter.setChildrenCollapsible(False)
        views_splitter.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        views_splitter.addWidget(self.profile_view)
        views_splitter.addWidget(self.parabola_view)
        views_splitter.setStretchFactor(0, PROFILE_VIEW_STRETCH)
        views_splitter.setStretchFactor(1, PARABOLA_VIEW_STRETCH)
        layout.addWidget(views_splitter, 1)
        # Same scale and x range, so aligned points line up on screen
        self.group = [self.profile_view, self.parabola_view]
        for view in self.group:
            view.group = self.group
            view.zoom_callback = self.zoom_views

        # Velocity triangles in a column per row on the right; beyond
        # ``MAX_VISIBLE_ROWS`` the area keeps its width and scrolls horizontally
        triangle_container = QWidget()
        self._triangle_layout = QHBoxLayout(triangle_container)
        self._triangle_layout.setContentsMargins(0, 0, 0, 0)
        self._triangle_layout.setSpacing(ROW_GAP)
        self._triangle_scroll = triangle_scroll = QScrollArea()
        triangle_scroll.setWidgetResizable(True)
        triangle_scroll.setFrameShape(QFrame.Shape.NoFrame)
        triangle_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        triangle_scroll.setWidget(triangle_container)
        self._root.addWidget(triangle_scroll)

        # Any geometry change schedules one Newton solve; changes arriving while it
        # is pending are merged, so dragging does not queue up solves
        self._solve_timer = QTimer(self)
        self._solve_timer.setSingleShot(True)
        self._solve_timer.setInterval(SOLVE_DELAY_MS)
        self._solve_timer.timeout.connect(self.update_solution)
        self._notifiers: list[ChangeNotifier] = []
        self._syncing = False  # leading edges are being redrawn from the solution

        self._append_row()
        self._fit_triangle_area()
        self._update_triangles()

        # Animation runs on the log of the zoom so steps compose smoothly
        self._zoom_applied = 0.0
        self._zoom_target = 0.0
        self._zoom_anim = QVariantAnimation(self)
        self._zoom_anim.setDuration(ZOOM_DURATION_MS)
        self._zoom_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._zoom_anim.valueChanged.connect(self._on_zoom_step)

        # Gliding fit of the views when a row is added
        self._fit_from: tuple[float, float, list[float]] = (1.0, 0.0, [])
        self._fit_to: tuple[float, float, list[float]] = (1.0, 0.0, [])
        self._fit_anim = QVariantAnimation(self)
        self._fit_anim.setStartValue(0.0)
        self._fit_anim.setEndValue(1.0)
        self._fit_anim.setDuration(FIT_DURATION_MS)
        self._fit_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._fit_anim.valueChanged.connect(self._on_fit_step)

        # Ctrl + / Ctrl - zoom both views together
        for keys, factor in (
            (('Ctrl++', 'Ctrl+='), ZOOM_STEP),
            (('Ctrl+-',), 1 / ZOOM_STEP),
        ):
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(lambda f=factor: self.zoom_views(f))
        # Ctrl 0 fits both views, like the button
        QShortcut(QKeySequence('Ctrl+0'), self).activated.connect(
            lambda: self.fit_views()
        )

    def _append_row(self):
        """Draw the last row of the backend after the rows already shown."""
        # A new row starts on the shaft of the last one, as its speed is the same; the
        # first row starts on the stationary casing
        shaft = self.rows[-1].shaft if self.rows else self.casing_shaft
        row = BladeRowView(
            self.backend,
            len(self.rows),
            self.rows[-1] if self.rows else None,
            self.profile_scene,
            self.parabola_scene,
            self.shaft_panel.shafts,
            shaft,
        )
        self.rows.append(row)
        row.shaft_combo.currentIndexChanged.connect(self._schedule_solve)

        # Shaft choice on top of the triangles of the row; all triangles share one scale
        column = QVBoxLayout()
        column.addWidget(row.shaft_combo)
        for triangle_view in row.triangle_views:
            # Columns keep their width, so extra rows scroll instead of squeezing
            triangle_view.setMinimumWidth(TRIANGLE_COLUMN_WIDTH - ROW_GAP)
            column.addWidget(triangle_view)
        self._triangle_layout.addLayout(column)
        self._row_ui.append(column)
        self._share_triangle_scale()

        self._notifiers.append(ChangeNotifier(row.points, self._schedule_solve))
        self.row_controls.attach(
            row.profile.center2, row.profile.end2, can_delete=len(self.rows) > 1
        )

    def _fit_triangle_area(self):
        """Size the triangle area for the rows shown; the others scroll."""
        shown = min(len(self.rows), MAX_VISIBLE_ROWS)
        self._root.setStretchFactor(2, shown)
        self._triangle_scroll.setMinimumWidth(shown * TRIANGLE_COLUMN_WIDTH)

    def _share_triangle_scale(self):
        views = self._triangle_views()
        for triangle_view in views:
            triangle_view.group = views

    def _triangle_views(self) -> list[VelocityTriangleView]:
        return [view for row in self.rows for view in row.triangle_views]

    def add_row(self):
        """Add a row after the last one, matched to the flow leaving the last row."""
        # Solve the current drawing first, so the new row starts from what is on screen
        self._solve_timer.stop()
        if not self.update_solution():
            self.status_label.setText('Fix the current solution before adding a row')
            return
        try:
            self.backend.add_row(self.backend.next_row_params())
        except RuntimeError as err:
            logger.warning(f'Could not add a row: {err}')
            self.status_label.setText('Could not solve with the added row')
            return
        self._append_row()
        self._update_triangles()
        self._fit_triangle_area()
        if len(self.rows) <= MAX_VISIBLE_ROWS:
            self.resize(self.width() + TRIANGLE_COLUMN_WIDTH, self.height())
        # The solve, the new widgets and the window resize all happen before this
        # point; let them settle, so the fit animation does not start with a stall
        QApplication.processEvents()
        self.profile_view._update_extent()
        self.fit_views()
        self.status_label.setText(f'Row {len(self.rows)} added and converged')

    def delete_row(self):
        """Remove the last row. The first row always stays."""
        if len(self.rows) < 2:
            return
        self._solve_timer.stop()
        # The remaining rows keep what is drawn, so it goes to the backend first
        values: dict[VarSpec, float] = {n0.oth.TotMassFlow: self.mass_flow_spin.value()}
        for row in self.rows[:-1]:
            values |= row.drawn_geometry() | row.operating_conditions()
        self.backend.set_geometry(values)
        try:
            self.backend.remove_last_row()
        except RuntimeError as err:
            logger.warning(f'Could not remove a row: {err}')
            self.status_label.setText('Could not solve without the last row')
            return

        self._complete_delete()

    def _complete_delete(self):
        """Take the vanished last row out of the window and fit the views."""
        row = self.rows.pop()
        row.remove(self._notifiers.pop())
        column = self._row_ui.pop()
        for widget in (row.shaft_combo, *row.triangle_views):
            column.removeWidget(widget)
            widget.deleteLater()
        self._triangle_layout.removeItem(column)
        column.deleteLater()
        self._share_triangle_scale()

        self.row_controls.attach(
            self.rows[-1].profile.center2,
            self.rows[-1].profile.end2,
            can_delete=len(self.rows) > 1,
        )
        self._update_triangles()
        self._fit_triangle_area()
        if len(self.rows) < MAX_VISIBLE_ROWS:
            self.resize(self.width() - TRIANGLE_COLUMN_WIDTH, self.height())
        QApplication.processEvents()
        self.profile_view._update_extent()
        self.fit_views()
        self.status_label.setText(f'Row {len(self.rows) + 1} removed and converged')

    def drawn_geometry(self) -> dict[VarSpec, float]:
        """Geometry currently drawn, as boundary conditions of the rows (m, rad)."""
        geometry: dict[VarSpec, float] = {}
        for row in self.rows:
            geometry |= row.drawn_geometry()
        return geometry

    def operating_conditions(self) -> dict[VarSpec, float]:
        """Mass flow (kg/s) and rotational speeds (rad/s) from the input fields."""
        conditions = {n0.oth.TotMassFlow: self.mass_flow_spin.value()}
        for row in self.rows:
            conditions |= row.operating_conditions()
        return conditions

    def _on_shaft_added(self, shaft: ShaftView):
        for row in self.rows:
            row.add_shaft_option(shaft)

    def _on_shaft_removed(self, index: int, removed: ShaftView):
        # Rows that were on the removed shaft move to the first moving shaft
        for row in self.rows:
            row.remove_shaft_option(index, removed, self.rotor_shaft)
        self._schedule_solve()

    def _on_shaft_speed_changed(self):
        for row in self.rows:
            row.update_rotation()
        self._schedule_solve()

    def _schedule_solve(self):
        if not self._syncing and not self._solve_timer.isActive():
            self._solve_timer.start()

    def update_solution(self) -> bool:
        """Pass the drawn geometry to ``kn`` and re-solve the rows with Newton."""
        self.backend.set_geometry(self.drawn_geometry() | self.operating_conditions())
        converged = self.backend.solve()
        if converged:
            self.status_label.setText('Newton converged')
        else:
            self.status_label.setText('Newton failed')
        self._update_triangles(converged)
        if converged:
            self._sync_inlet_angles()
        return converged

    def _sync_inlet_angles(self):
        """Draw the leading edges of the following rows at the solved angle.

        Moving them is not an edit, so it must not trigger another solve.
        """
        self._syncing = True
        try:
            for row in self.rows:
                row.sync_inlet_angle()
        finally:
            self._syncing = False

    def _update_triangles(self, converged: bool = True):
        """Show the velocity triangles of the current solution."""
        for row in self.rows:
            row.update_triangles(converged)
        if converged:
            VelocityTriangleView.fit_group(self._triangle_views())

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

    def _fit_target(self) -> tuple[float, float, list[float]]:
        """Common scale, shared x centre and y centre per view that fit the content."""
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
        return min(scales), (left + right) / 2, [rect.center().y() for rect in rects]

    def _show_view(self, scale: float, center_x: float, center_ys: list[float]):
        for view, center_y in zip(self.group, center_ys):
            view.resetTransform()
            view.scale(scale, scale)
            view.centerOn(center_x, center_y)

    def _current_view(self) -> tuple[float, float, list[float]]:
        """Scale, shared x centre and y centre per view as shown now."""
        centers = []
        for view in self.group:
            viewport = view.viewport()
            assert viewport is not None
            centers.append(view.mapToScene(viewport.rect().center()))
        return (
            self.profile_view.transform().m11(),
            centers[0].x(),
            [center.y() for center in centers],
        )

    def fit_views(self, animated: bool = True):
        """Fit the profile and the parabola (not the axis) using one common scale.

        With ``animated`` (the default) the views glide to the fit with an ease.
        """
        self._zoom_anim.stop()
        self._fit_anim.stop()
        end = self._fit_target()
        if not animated:
            self._show_view(*end)
            return
        self._fit_from, self._fit_to = self._current_view(), end
        self._fit_anim.start()

    def _on_fit_step(self, progress):
        """Show the views a fraction ``progress`` of the way to the fit."""
        t = float(progress)
        (scale0, x0, ys0), (scale1, x1, ys1) = self._fit_from, self._fit_to
        # The scale is interpolated in its log, like the zoom, so it feels uniform
        scale = math.exp(math.log(scale0) + (math.log(scale1) - math.log(scale0)) * t)
        self._show_view(
            scale,
            x0 + (x1 - x0) * t,
            [y0 + (y1 - y0) * t for y0, y1 in zip(ys0, ys1)],
        )
