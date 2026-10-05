"""Views of the GUI: profile and camber views,
velocity triangles and the main window."""

import logging
import math
from collections.abc import Callable
from pathlib import Path

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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    GapFollower,
    MeridionalProfile,
    ParabolicLine,
)
from adet.gui.labels import AngleLabel, RadiusAxis, RotationArrow
from adet.gui.row_backend import FluidChoice, RowBackend, n0
from adet.variables import VarSpec

logger = logging.getLogger(__name__)

# *** Animations
ZOOM_STEP = 1.25
ZOOM_DURATION_MS = 150
FIT_DURATION_MS = 100  # glide of the views to the fit when a row is added

FIT_MARGIN = 20  # scene units around the content when fitting
CORNER_BUTTON_MARGIN = 8  # pixels between a view's corner button and its edges
CORNER_BUTTON_RIGHT_MARGIN = 16  # pixels between a corner button and the right edge
FIT_ICON_SIZE = 20  # pixels
FIT_BUTTON_SIZE = 33  # pixels
RADIUS_ORIGIN_Y = 550.0  # scene y of radius 0
SCENE_PER_METER = 2000.0  # the initial center at y=350 is then 0.1 m
PROFILE_START_X = 150.0  # scene x of the inlet station
PARABOLA_START_Y = 150.0  # scene y of the first point of the camber parabola
AXIAL_GAP = 20.0  # visual gap between blade rows along the wall, scene units
TRIANGLE_COLUMN_WIDTH = 350  # window growth per added row, pixels
MAX_VISIBLE_ROWS = 2  # rows whose triangles grow the window; more rows scroll
ROW_LABEL_FONT_PX = 28  # row number above each triangle column
ROW_GAP = 16  # spacing between the widgets of neighbouring blade rows, pixels
SOLVE_DELAY_MS = 30  # geometry changes within this window share one solve
BACKGROUND_COLOR = QColor(15, 15, 15)  # background of every view
CASING_COLOR = QColor(170, 170, 170)
DEFAULT_OMEGA = 100.0  # rad/s, first speed of a moving shaft
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
PANEL_BORDER_COLOR = QColor(80, 80, 80)  # enclosure of the fluid and shaft panels
SPLITTER_HANDLE_WIDTH = 10  # gap between views, wide enough to grab
PROFILE_VIEW_STRETCH = 3  # startup height share of the meridional profile view
PARABOLA_VIEW_STRETCH = 2  # startup height share of the camber line view
TIP_GRAB_RADIUS = 14  # pixels around the tip of V that grab the mouse
TIP_HANDLE_RADIUS = 5  # pixels
SWITCH_TRACK_SIZE = (44, 22)  # width, height of the sliding switch track, pixels
SWITCH_KNOB_MARGIN = 3  # gap between the knob and the track edge, pixels
SWITCH_SLIDE_MS = 120  # duration of the knob slide
SWITCH_TRACK_COLOR = QColor(80, 80, 80)
SWITCH_KNOB_COLOR = QColor(58, 134, 255)
SWITCH_ACTIVE_TEXT_COLOR = QColor(230, 230, 230)
SWITCH_INACTIVE_TEXT_COLOR = QColor(130, 130, 130)
MAX_FLOW_ANGLE = math.radians(80)  # dragged relative flow angles stay within +-this
FLOW_ANGLE_SNAP_DEG = 5.0  # step of the dragged angle while Ctrl is held


class SlideSwitch(QCheckBox):
    """Left/right switch whose knob slides between two labelled options.

    Unchecked is the left option and checked the right one. It keeps the ``QCheckBox``
    interface (``toggled``, ``setChecked``, ``setEnabled``).
    """

    def __init__(self, left_text: str, right_text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.left_text = left_text
        self.right_text = right_text
        self._position = 0.0  # 0 knob on the left, 1 on the right
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(SWITCH_SLIDE_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._animation.valueChanged.connect(self._set_position)
        self.toggled.connect(self._slide)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def setChecked(self, checked: bool):
        """Set the state; without a ``toggled`` signal the knob jumps there.

        Setting the state it already has leaves a running slide untouched.
        """
        if checked == self.isChecked():
            return
        super().setChecked(checked)
        self._animation.stop()
        self._position = 1.0 if checked else 0.0
        self.update()

    def sizeHint(self):
        metrics = self.fontMetrics()
        width = (
            metrics.horizontalAdvance(self.left_text)
            + metrics.horizontalAdvance(self.right_text)
            + SWITCH_TRACK_SIZE[0]
            + 4 * SWITCH_KNOB_MARGIN
        )
        return QRectF(0, 0, width, SWITCH_TRACK_SIZE[1] + 4).size().toSize()

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def _slide(self, checked: bool):
        self._animation.stop()
        self._animation.setStartValue(self._position)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def _set_position(self, position: float):
        self._position = position
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(1.0 if self.isEnabled() else 0.4)
        metrics = self.fontMetrics()
        track_w, track_h = SWITCH_TRACK_SIZE
        left_w = metrics.horizontalAdvance(self.left_text)
        track_x = left_w + 2 * SWITCH_KNOB_MARGIN
        track_y = (self.height() - track_h) / 2
        height = self.height()

        # Option labels, the selected one highlighted
        painter.setPen(
            SWITCH_INACTIVE_TEXT_COLOR if self.isChecked() else SWITCH_ACTIVE_TEXT_COLOR
        )
        painter.drawText(
            QRectF(0, 0, left_w, height), Qt.AlignmentFlag.AlignVCenter, self.left_text
        )
        painter.setPen(
            SWITCH_ACTIVE_TEXT_COLOR if self.isChecked() else SWITCH_INACTIVE_TEXT_COLOR
        )
        painter.drawText(
            QRectF(track_x + track_w + 2 * SWITCH_KNOB_MARGIN, 0, self.width(), height),
            Qt.AlignmentFlag.AlignVCenter,
            self.right_text,
        )

        # Track and knob
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(SWITCH_TRACK_COLOR)
        painter.drawRoundedRect(
            QRectF(track_x, track_y, track_w, track_h), track_h / 2, track_h / 2
        )
        knob = track_h - 2 * SWITCH_KNOB_MARGIN
        knob_x = track_x + SWITCH_KNOB_MARGIN
        knob_x += self._position * (track_w - knob - 2 * SWITCH_KNOB_MARGIN)
        painter.setBrush(SWITCH_KNOB_COLOR)
        painter.drawEllipse(QRectF(knob_x, track_y + SWITCH_KNOB_MARGIN, knob, knob))


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
    height = 2 * math.hypot(dx, dy_up) / SCENE_PER_METER
    # A horizontal line (radial station) keeps the side its end is on: +90 degrees to
    # the right, -90 to the left. The sign is the flow direction (see ``_line_points``,
    # the normal is (cos, sin) in the scene), so the rounding noise in y is dropped
    if abs(dy_up) <= 1e-9 * max(1.0, abs(dx)):
        return height, math.copysign(math.pi / 2, dx)
    # Either end can be the upper one; fold the direction into the upper half plane
    if dy_up < 0:
        dx, dy_up = -dx, -dy_up
    return height, math.atan2(dx, dy_up)


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


def _enclose(panel: QWidget, name: str):
    """Draw a rounded rectangle border around ``panel``; its children are unaffected."""
    panel.setObjectName(name)
    panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    panel.setStyleSheet(
        f'#{name} {{ border: 1px solid {PANEL_BORDER_COLOR.name()};'
        f' border-radius: {int(VIEW_CORNER_RADIUS)}px; }}'
    )


class StatusLabel(QLabel):
    """Wrapping status message, green normally and red when the text reports a failure."""

    FAILURE_MARKERS = ('could not', 'failed', 'fix the current')

    def setText(self, text: str | None):
        failed = any(marker in (text or '').lower() for marker in self.FAILURE_MARKERS)
        color = 'red' if failed else 'green'
        self.setStyleSheet(f'color: {color}; font-weight: bold;')
        super().setText(text)


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
        _enclose(self, 'shaftPanel')
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(ROW_GAP)
        title = QLabel('Shafts')
        title.setStyleSheet('font-weight: bold; font-size: 15px;')
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
        # A stationary shaft would divide by zero in the coefficients of its rows
        omega = next(
            (shaft.omega for shaft in reversed(self.shafts) if shaft.omega != 0.0),
            DEFAULT_OMEGA,
        )
        return self.add_shaft(name, color, omega, removable=True)

    def remove_shaft(self, shaft: ShaftView):
        """Take a shaft out of the panel and tell the listeners."""
        index = self.shafts.index(shaft)
        self.shafts.pop(index)
        self._layout.removeWidget(shaft)
        shaft.hide()  # until the deferred delete happens
        shaft.deleteLater()
        self.shaft_removed.emit(index, shaft)


class FluidPanel(QWidget):
    """Working fluid choice: an ideal gas or a CoolProp fluid.

    ``changed`` fires with the new ``FluidChoice`` when the user applies it (Enter in
    a field, a spin box edit or the model switch).
    """

    changed = pyqtSignal(FluidChoice)

    def __init__(self, fluid: FluidChoice):
        super().__init__()
        self.model_combo = QComboBox()
        self.model_combo.addItems(['Ideal', 'CoolProp'])
        self.gamma_spin = _make_spin(fluid.gamma, 1.0001, 3.0, 0.05, 3)
        self.gas_constant_spin = _make_spin(fluid.gas_constant, 1e-3, 1e5, 1.0, 3)
        self.viscosity_spin = _make_spin(fluid.viscosity * 1e6, 1e-3, 1e6, 1.0, 3)
        self.name_edit = QLineEdit(fluid.name)
        self.backend_edit = QLineEdit(fluid.backend)

        _enclose(self, 'fluidPanel')
        layout = QVBoxLayout(self)
        layout.setSpacing(ROW_GAP // 2)
        title = QLabel('Working fluid')
        title.setStyleSheet('font-weight: bold; font-size: 15px;')
        layout.addWidget(title)
        layout.addWidget(self.model_combo)
        self._ideal_fields = self._add_fields(
            layout,
            (
                ('gamma [-]', self.gamma_spin),
                ('R [J/(kg K)]', self.gas_constant_spin),
                ('mu [uPa s]', self.viscosity_spin),
            ),
        )
        self._coolprop_fields = self._add_fields(
            layout, (('Fluid', self.name_edit), ('Backend', self.backend_edit))
        )
        self.model_combo.setCurrentIndex(0 if fluid.ideal else 1)
        self._show_fields()
        self._last = fluid

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        for spin in (self.gamma_spin, self.gas_constant_spin, self.viscosity_spin):
            spin.valueChanged.connect(self._emit)
        for edit in (self.name_edit, self.backend_edit):
            edit.editingFinished.connect(self._emit)

    @staticmethod
    def _add_fields(layout: QVBoxLayout, fields) -> QWidget:
        """Labelled ``fields`` in one widget, so they show and hide together."""
        container = QWidget()
        inner = QVBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        for text, widget in fields:
            row = QHBoxLayout()
            row.addWidget(QLabel(text))
            row.addWidget(widget, 1)
            inner.addLayout(row)
        layout.addWidget(container)
        return container

    def _show_fields(self):
        ideal = self.model_combo.currentIndex() == 0
        self._ideal_fields.setVisible(ideal)
        self._coolprop_fields.setVisible(not ideal)

    def _on_model_changed(self):
        self._show_fields()
        self._emit()

    @property
    def fluid(self) -> FluidChoice:
        return FluidChoice(
            ideal=self.model_combo.currentIndex() == 0,
            gamma=self.gamma_spin.value(),
            gas_constant=self.gas_constant_spin.value(),
            viscosity=self.viscosity_spin.value() * 1e-6,
            name=self.name_edit.text().strip(),
            backend=self.backend_edit.text().strip(),
        )

    def set_fluid(self, fluid: FluidChoice):
        """Show ``fluid`` again (after a failed change) without notifying."""
        self._last = fluid  # so the edits below are not seen as changes
        widgets = self.findChildren(QWidget)
        for widget in widgets:
            widget.blockSignals(True)
        self.model_combo.setCurrentIndex(0 if fluid.ideal else 1)
        self.gamma_spin.setValue(fluid.gamma)
        self.gas_constant_spin.setValue(fluid.gas_constant)
        self.viscosity_spin.setValue(fluid.viscosity * 1e6)
        self.name_edit.setText(fluid.name)
        self.backend_edit.setText(fluid.backend)
        for widget in widgets:
            widget.blockSignals(False)
        self._show_fields()

    def _emit(self):
        fluid = self.fluid
        # Editing also finishes on focus loss without a change
        if fluid != self._last:
            self._last = fluid
            self.changed.emit(fluid)


class InletPanel(QWidget):
    """Inlet total conditions: total pressure (bar) and total temperature (K).

    ``changed`` fires on any edit.
    """

    changed = pyqtSignal()

    def __init__(self, pressure: float, temperature: float):
        super().__init__()
        self.pressure_spin = _make_spin(pressure * 1e-5, 1e-3, 1e4, 0.1, 3)
        self.temperature_spin = _make_spin(temperature, 1.0, 1e4, 5.0, 2)

        _enclose(self, 'inletPanel')
        layout = QVBoxLayout(self)
        layout.setSpacing(ROW_GAP // 2)
        title = QLabel('Inlet conditions')
        title.setStyleSheet('font-weight: bold; font-size: 15px;')
        layout.addWidget(title)
        for text, spin in (
            ('Total pressure [bar]', self.pressure_spin),
            ('Total temperature [K]', self.temperature_spin),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(text))
            row.addWidget(spin, 1)
            layout.addLayout(row)
            spin.valueChanged.connect(self.changed)

    @property
    def pressure(self) -> float:
        """Total pressure, Pa."""
        return self.pressure_spin.value() * 1e5

    @property
    def temperature(self) -> float:
        """Total temperature, K."""
        return self.temperature_spin.value()

    def set_conditions(self, pressure: float, temperature: float):
        """Show the total pressure (Pa) and temperature (K) without notifying."""
        for spin, value in (
            (self.pressure_spin, pressure * 1e-5),
            (self.temperature_spin, temperature),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)


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


def _fit_icon() -> QIcon:
    """Two arrows along the diagonal pointing away from each other."""
    size = FIT_ICON_SIZE
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor('#ffffff'), 1.6))
    low, high, head = 2.5, size - 2.5, 5.0
    painter.drawLine(QPointF(low, high), QPointF(high, low))
    for tip, sign in ((QPointF(low, high), 1), (QPointF(high, low), -1)):
        painter.drawLine(tip, QPointF(tip.x() + sign * head, tip.y()))
        painter.drawLine(tip, QPointF(tip.x(), tip.y() - sign * head))
    painter.end()
    return QIcon(pixmap)


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
    """A fixed-scale view that can share its x range and scroll position with a group.

    Views in the same group always have the same x range and horizontal scroll
    position. By default a view is alone in its group.
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
        self._corner_button: QWidget | None = None

    def set_corner_button(self, button: QWidget):
        """Float ``button`` over the top right corner of the view."""
        button.setParent(self)
        self._corner_button = button
        self._place_corner_button()
        button.show()

    def _place_corner_button(self):
        if self._corner_button is not None:
            size = self._corner_button.sizeHint()
            self._corner_button.setGeometry(
                self.width() - size.width() - CORNER_BUTTON_RIGHT_MARGIN,
                CORNER_BUTTON_MARGIN,
                size.width(),
                size.height(),
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_corner_button()

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


class TriangleGraphicsView(RoundedGraphicsView):
    """Graphics view of a velocity triangle whose W tip can be dragged vertically.

    The meridional velocity is set by the mass flow, so the tip only moves along the
    tangential direction: ``on_angle_dragged`` gets the relative flow angle (rad) of
    the tip at the cursor height and the current meridional velocity. Without a
    callback the tip is not draggable.
    """

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.tip: QPointF | None = None  # tip of W in scene coordinates
        self.on_angle_dragged: Callable[[float], None] | None = None
        self._dragging = False
        viewport = self.viewport()
        assert viewport is not None
        viewport.setMouseTracking(True)  # hover feedback without a button pressed

    def _over_tip(self, pos: QPointF) -> bool:
        if self.tip is None or self.on_angle_dragged is None:
            return False
        tip = self.mapFromScene(self.tip)
        return math.hypot(pos.x() - tip.x(), pos.y() - tip.y()) <= TIP_GRAB_RADIUS

    def _drag_to(self, event):
        """Report the flow angle at the cursor height, snapped while Ctrl is held."""
        if self.tip is None:
            return
        cursor = self.mapToScene(event.position().toPoint())
        # Scene y points down, so the tangential velocity is the negated scene y; the
        # meridional velocity (scene x) does not follow the cursor
        angle = math.atan2(-cursor.y(), self.tip.x())
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            step = math.radians(FLOW_ANGLE_SNAP_DEG)
            angle = round(angle / step) * step
        angle = max(-MAX_FLOW_ANGLE, min(MAX_FLOW_ANGLE, angle))
        if self.on_angle_dragged is not None:
            self.on_angle_dragged(angle)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._over_tip(
            event.position()
        ):
            self._dragging = True
            self._drag_to(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        viewport = self.viewport()
        assert viewport is not None
        if self._dragging:
            self._drag_to(event)
            event.accept()
            return
        viewport.setCursor(
            Qt.CursorShape.PointingHandCursor
            if self._over_tip(event.position())
            else Qt.CursorShape.ArrowCursor
        )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class VelocityTriangleView(QWidget):
    """Velocity triangle of one station, drawn with Qt graphics items.

    Axes follow ``plot_velocity_triangles``: V_m to the right, V_t upwards. W and V
    start at the origin and U closes the triangle. With ``on_angle_dragged`` set the
    tip of W can be dragged along V_t to impose the relative flow angle of the station.
    """

    ARROW_HEAD = 0.06  # head length as a fraction of the drawn extent
    COLORS = {
        'W': QColor(40, 110, 220),
        'U': QColor(150, 80, 200),
        'V': QColor(220, 60, 50),
    }

    def __init__(self, title: str):
        super().__init__()
        self.absolute = False  # imposed angle is the relative one (tip of W)
        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(QBrush(BACKGROUND_COLOR))
        self._view = TriangleGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        # The visible range is set exactly by ``fit_group``, so no scrolling
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.data_rect = QRectF()  # area the drawing needs, in scene units
        self.group: list[VelocityTriangleView] = [self]  # views sharing scale and x
        self._axis_items: list[QGraphicsItem] = []
        # Velocities on screen (v_tan, v_mer, U)
        self._shown: tuple[float, float, float] | None = None
        self._readout = QLabel('')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Not in the layout: the owner places the title above the angle switch
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet('font-weight: bold;')
        layout.addWidget(self._view, 1)
        layout.addWidget(self._readout)

    @property
    def on_angle_dragged(self) -> Callable[[float], None] | None:
        return self._view.on_angle_dragged

    @on_angle_dragged.setter
    def on_angle_dragged(self, callback: Callable[[float], None] | None):
        self._view.on_angle_dragged = callback

    def set_absolute(self, absolute: bool):
        """Drag the tip of V (absolute angle) instead of the tip of W (relative)."""
        self.absolute = absolute
        if self._shown is not None:
            self._draw(*self._shown)

    def set_title_color(self, color: QColor):
        self.title_label.setStyleSheet(f'font-weight: bold; color: {color.name()};')

    def set_velocities(self, v_tan: float, v_mer: float, blade_speed: float):
        """Draw the triangle with the new velocities (m/s)."""
        first = self._shown is None
        self._draw(v_tan, v_mer, blade_speed)
        if not first:
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
        # The tip that is dragged is the one of the imposed angle
        drag_tip, drag_name = (v_tip, 'V') if self.absolute else (w_tip, 'W')
        self._view.tip = drag_tip
        if self.on_angle_dragged is not None:
            handle = QGraphicsEllipseItem(
                -TIP_HANDLE_RADIUS,
                -TIP_HANDLE_RADIUS,
                2 * TIP_HANDLE_RADIUS,
                2 * TIP_HANDLE_RADIUS,
            )
            handle.setBrush(QBrush(self.COLORS[drag_name]))
            handle.setPen(QPen(QColor('white'), 1))
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            handle.setPos(drag_tip)
            handle.setZValue(1)
            self._scene.addItem(handle)

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
        angle = math.degrees(math.atan2(v_tan if self.absolute else w_tan, v_mer))
        self._readout.setText(
            f'{"α" if self.absolute else "β"} = {angle:.1f}°   Vm = {v_mer:.1f}   Vt = {v_tan:.1f}   '
            f'Wt = {w_tan:.1f}   U = {blade_speed:.1f} m/s'
        )

    def set_failed(self):
        """Remove the triangle and warn that the solution failed."""
        self._shown = None
        self._view.tip = None
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

    The relative flow angles are imposed by dragging the tip of W in the outlet
    triangle (and in the inlet triangle of the first row), which calls
    ``on_flow_changed``. The camber line is a result of the solution.
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
        on_flow_changed: Callable[[], None],
        on_angle_mode_changed: Callable[['BladeRowView', int, bool], None],
    ):
        self.backend = backend
        self.on_flow_changed = on_flow_changed
        self.on_angle_mode_changed = on_angle_mode_changed
        self.index = index
        self.previous = previous
        self.shafts = shafts
        self.shaft = shaft
        self.profile_scene = profile_scene
        self.parabola_scene = parabola_scene
        self.inlet, self.outlet = backend.row_nodes(index)
        get = backend.get_value

        # Meridional profile in the profile scene
        axial_length = get(self.outlet.geo.AxialLength) * SCENE_PER_METER
        # A following row is drawn after a visual gap, which the solution ignores
        self.gap = 0.0 if previous is None else AXIAL_GAP
        # The gap is along the meridional direction, which is not the x axis when the
        # station line is tilted (and is the y axis for a radial row)
        if previous is None:
            start_x = PROFILE_START_X
            start_y = RADIUS_ORIGIN_Y - get(self.inlet.geo.Rmid) * SCENE_PER_METER
        else:
            prev_profile = previous.profile
            shift = GapFollower.offset(
                prev_profile.center2, prev_profile.end2, self.gap
            )
            start_x = prev_profile.center2.pos().x() + shift.x()
            start_y = prev_profile.center2.pos().y() + shift.y()
        # The axial length is what x advances between the centers (a radial row has
        # almost none). The visual gap does not count: the solution has none
        # The outlet line must not reach back over the inlet one: a radial row has a
        # horizontal outlet line, which centered on the axial advance would cross the
        # previous row. Keep its leftmost point at the rightmost point of the inlet line.
        # The signed half widths make parallel lines (a radial row after a radial
        # row) need no advance, so that the row goes straight on
        inlet_half_x = (
            get(self.inlet.geo.Height)
            / 2
            * SCENE_PER_METER
            * math.sin(get(self.inlet.geo.MeridionalAngle))
        )
        outlet_half_x = (
            get(self.outlet.geo.Height)
            / 2
            * SCENE_PER_METER
            * math.sin(get(self.outlet.geo.MeridionalAngle))
        )
        axial_length = max(axial_length, abs(outlet_half_x - inlet_half_x))
        # The row extends from its leading edge by the radius change of the solution.
        # The leading edge of a following row is moved by the visual gap, which the
        # solution ignores, so the drawn radius is not the solved one
        drawn_inlet_radius = (RADIUS_ORIGIN_Y - start_y) / SCENE_PER_METER
        center2, end2 = _line_points(
            start_x + axial_length,
            drawn_inlet_radius + get(self.outlet.geo.Rmid) - get(self.inlet.geo.Rmid),
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

        # Parabola in the parabola scene, independent of the profile view. It spans
        # the meridional chord (from the axial length through MinimalMeridional), which
        # is the extent of the blade-to-blade plane along the meridional direction
        # The camber parabola is drawn from the solved metal angles (positive = rising)
        metal0 = get(self.inlet.geo.MetalAngle)
        metal1 = get(self.outlet.geo.MetalAngle)
        camber_chord = get(self.outlet.geo.MerChord) * SCENE_PER_METER
        half_dx = camber_chord / 2
        # A following row starts where the previous camber line ends
        start_x, start_y = (
            (PROFILE_START_X, PARABOLA_START_Y)
            if previous is None
            else (
                previous.camber_points[2].pos().x() + self.gap,
                previous.camber_points[2].pos().y(),
            )
        )
        control_y = start_y - half_dx * math.tan(metal0)
        end_y = control_y - half_dx * math.tan(metal1)
        start = DraggablePoint(start_x, start_y)
        end = DraggablePoint(start_x + camber_chord, end_y)
        control = DraggablePoint(start_x + half_dx, control_y)
        self.parabola = ParabolicLine(start, control, end)
        for item in (start, end, control, self.parabola):
            parabola_scene.addItem(item)
        self.camber_points = (start, control, end)
        self.camber_alignment: AlignedPoints | None = None
        if previous is not None:
            # The leading edge stays at the height of the previous trailing edge
            self.camber_alignment = AlignedPoints(previous.camber_points[2], start, 'x')
        # The camber line is a result of the solution (``sync_camber``), so none of
        # its points is draggable: only the line is shown
        for point in self.camber_points:
            point.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            point.setVisible(False)

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
        # Coefficients of the row, shown only while it turns
        self.coefficients_label = QLabel('')
        self.coefficients_label.setVisible(False)

        # Inlet and outlet velocity triangles
        self.triangle_views = (
            VelocityTriangleView('Inlet'),
            VelocityTriangleView('Outlet'),
        )
        # Row number on top of the column
        self.row_label = QLabel(f'Row {index + 1}')
        self.row_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Choice of the flow angle that is imposed by dragging, per triangle. The inlet
        # of a following row takes the flow from the previous row, so it has no choice
        self.angle_switches = (
            SlideSwitch('Relative angle', 'Absolute angle'),
            SlideSwitch('Relative angle', 'Absolute angle'),
        )
        for node, switch in enumerate(self.angle_switches):
            # The change solves synchronously,
            # so it waits for the knob to finish sliding
            switch.toggled.connect(
                lambda checked, node=node: QTimer.singleShot(
                    SWITCH_SLIDE_MS,
                    lambda: self.on_angle_mode_changed(self, node, checked),
                )
            )
        self.angle_switches[0].setEnabled(previous is None)
        self.flow_angles: dict[VarSpec, float] = {}
        self.set_angle_mode()
        self._apply_shaft_color()

    def set_angle_mode(self):
        """Impose the flow angles of the kinds the backend uses, from its solution.

        The inlet of a following row takes the flow from the previous row, so its
        angle is a result and not imposed.
        """
        self.flow_angles = {}
        for node, (view, switch) in enumerate(
            zip(self.triangle_views, self.angle_switches)
        ):
            absolute = self.backend.is_absolute(self.index, node)
            if node == 1 or self.previous is None:
                spec = self.backend.angle_spec(self.index, node)
                self.flow_angles[spec] = self.backend.get_value(spec)
                view.on_angle_dragged = lambda angle, spec=spec: self._set_flow_angle(
                    spec, angle
                )
            view.set_absolute(absolute)
            switch.blockSignals(True)
            switch.setChecked(absolute)
            switch.blockSignals(False)

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
        self.row_label.setStyleSheet(
            f'font-weight: bold; font-size: {ROW_LABEL_FONT_PX}px; color: {color.name()};'
        )
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
            *self.profile.perpendiculars,
            self.profile.spline_a,
            self.profile.spline_b,
            self.profile.fill,
        ]
        if self.profile.gap_follower is not None:
            owners.append(self.profile.gap_follower)
        root = self.profile.root
        if self.profile in root.chained:
            root.chained.remove(self.profile)
        shared = self.previous.profile
        for point in (shared.center2, shared.line2.end1, shared.line2.end2):
            for owner in owners:
                point.remove_dependent(owner)
        if self.camber_alignment is not None:
            self.previous.camber_points[2].remove_dependent(self.camber_alignment)
        for item in (
            *self.profile.items,
            self.profile.snap_highlight,
            *self.camber_points,
            self.parabola,
            self.inlet_label,
            self.outlet_label,
            self.rotation_arrow,
        ):
            scene = item.scene()
            if scene is not None:
                # The lines are drawn from their points, so their bounds must be
                # invalidated while those still have their position; otherwise the
                # removed line can stay painted
                scene.update(item.sceneBoundingRect().adjusted(-4, -4, 4, 4))
                scene.removeItem(item)
        for scene in (self.profile_scene, self.parabola_scene):
            scene.update()
            for view in scene.views():
                if (viewport := view.viewport()) is not None:
                    viewport.update()

    @property
    def points(self) -> tuple[DraggablePoint, ...]:
        """Every point whose movement changes the geometry of the row."""
        profile = self.profile
        return (profile.center1, profile.end1, profile.center2, profile.end2)

    def _set_flow_angle(self, spec: VarSpec, angle: float):
        """Impose a new relative flow angle (rad) and ask for a solve."""
        if self.flow_angles[spec] != angle:
            self.flow_angles[spec] = angle
            self.on_flow_changed()

    def _solved_outlet_radius(self) -> float:
        """Radius (m) of the outlet center as the solution sees it.

        The row is as long as it is drawn from its leading edge to its trailing edge,
        and starts at the outlet radius of the previous row, so the visual gap between
        the rows does not change the radius.
        """
        if self.previous is None:
            inlet_radius = (RADIUS_ORIGIN_Y - self.profile.center1.pos().y()) / (
                SCENE_PER_METER
            )
        else:
            inlet_radius = self.previous._solved_outlet_radius()
        drawn_change = self.profile.center1.pos().y() - self.profile.center2.pos().y()
        return inlet_radius + drawn_change / SCENE_PER_METER

    def drawn_geometry(self) -> dict[VarSpec, float]:
        """Geometry currently drawn, as boundary conditions of the row (m, rad).

        The inlet geometry of a row that follows another one comes from the link.
        """
        profile = self.profile
        center1 = profile.center1.get_position()
        center2 = profile.center2.get_position()
        height1, mer_angle1 = _line_geometry(center2, profile.end2.get_position())
        geometry = {
            self.outlet.geo.Rmid: self._solved_outlet_radius(),
            self.outlet.geo.Height: height1,
            self.outlet.geo.MeridionalAngle: mer_angle1,
            # The gap lies before center1, so it is not part of the axial length, which
            # is the x distance between the centers of the leading and trailing edges
            self.outlet.geo.AxialLength: (center2.x() - center1.x()) / SCENE_PER_METER,
        }
        if self.radius_axis is not None:
            height0, mer_angle0 = _line_geometry(center1, profile.end1.get_position())
            geometry |= {
                self.inlet.geo.Rmid: self.radius_axis.radius(),
                self.inlet.geo.Height: height0,
                self.inlet.geo.MeridionalAngle: mer_angle0,
            }
        return geometry

    def sync_camber(self):
        """Draw the camber line of the solved metal angles.

        The leading edge is pinned in y (to the previous trailing edge for a following
        row), so the control point and then the trailing edge move to the tangents.
        The trailing edge drags the leading edge of the next row along.
        """
        start, control, end = self.camber_points
        get = self.backend.get_value
        chord = get(self.outlet.geo.MerChord) * SCENE_PER_METER
        half_dx = chord / 2  # control sits at the midpoint of the meridional chord
        # The leading edge follows the trailing edge of the previous row in x too,
        # keeping the visual gap between the rows
        start_x = start.pos().x()
        if self.previous is not None:
            start_x = self.previous.camber_points[2].pos().x() + self.gap
            # The start point is constrained to move in y when dragged, which would
            # also ignore this x move and let the camber lines overlap
            start.axis_constraint = None
            start.setPos(start_x, start.pos().y())
            start.axis_constraint = 'y'
        # The trailing edge can never go back over the leading edge
        end.min_x = start_x
        control_y = start.pos().y() - half_dx * math.tan(get(self.inlet.geo.MetalAngle))
        control.setPos(start_x + half_dx, control_y)
        end_y = control_y - half_dx * math.tan(get(self.outlet.geo.MetalAngle))
        end.setPos(start_x + chord, end_y)

    def operating_conditions(self) -> dict[VarSpec, float]:
        """Rotational speed (rad/s) of the shaft and imposed relative flow angles."""
        return {self.outlet.kin.Omega: self.shaft.omega} | self.flow_angles

    def update_triangles(self, converged: bool = True):
        """Show the velocity triangles of the current solution."""
        if not converged:
            for triangle_view in self.triangle_views:
                triangle_view.set_failed()
            self.coefficients_label.setVisible(False)
            return
        get = self.backend.get_value
        for triangle_view, node in zip(self.triangle_views, (self.inlet, self.outlet)):
            triangle_view.set_velocities(
                get(node.kin.V_tan), get(node.kin.V_mer), get(node.kin.BladeSpeed)
            )
        self.update_coefficients()

    def update_coefficients(self):
        """Show the work and flow coefficients and the efficiency of a rotor.

        They divide by the blade speed, so a row on a stationary shaft shows none.
        """
        if self.shaft.omega == 0.0:
            self.coefficients_label.setVisible(False)
            return
        work, flow, eta = self.backend.rotor_coefficients(self.index)
        self.coefficients_label.setText(
            f'ψ = {work:.3f}   φ = {flow:.3f}   η_tt = {eta:.3f}'
        )
        self.coefficients_label.setVisible(True)


class MainGuiView(QWidget):
    """Chain of blade rows in two shared views, so lines never overlap.

    The meridional profiles of all rows are in one view and the camber lines in
    another. Each row has its own pair of velocity triangles. The ``+`` button adds
    a row after the last one.
    """

    _windows: list['MainGuiView'] = []  # imported windows, kept from garbage collection

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
        fit_button = QPushButton()
        fit_button.setIcon(_fit_icon())
        fit_button.setToolTip('Fit both views (Ctrl+0)')
        fit_button.setFixedSize(FIT_BUTTON_SIZE, FIT_BUTTON_SIZE)
        fit_button.clicked.connect(lambda _checked=False: self.fit_views())
        self.profile_view.set_corner_button(fit_button)
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
            DEFAULT_OMEGA,
        )
        self.fluid_panel = FluidPanel(self.backend.fluid)
        self.fluid_panel.changed.connect(self._on_fluid_changed)
        self.inlet_panel = InletPanel(
            self.backend.get_value(n0.tot.Pressure),
            self.backend.get_value(n0.tot.Temperature),
        )
        self.inlet_panel.changed.connect(self._schedule_solve)
        self.shaft_panel.changed.connect(self._on_shaft_speed_changed)
        self.shaft_panel.shaft_added.connect(self._on_shaft_added)
        self.shaft_panel.shaft_removed.connect(self._on_shaft_removed)

        # Panels are separated by draggable splitter handles
        self._root = QSplitter(Qt.Orientation.Horizontal)
        root_layout = QHBoxLayout(self)
        root_layout.addWidget(self._root)
        # Fluid and inlet conditions above the shafts, in one column
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.fluid_panel)
        left_layout.addWidget(self.inlet_panel)
        left_layout.addWidget(self.shaft_panel, 1)
        self._root.addWidget(left_column)
        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        self._root.addWidget(center)
        self._root.setStretchFactor(1, 3)
        self._root.setChildrenCollapsible(False)
        self._root.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        # Button row on the top right
        top_row = QHBoxLayout()
        self.status_label = StatusLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setText('Initial solution converged')
        # Manual recovery with Ipopt, only offered after Newton failed
        self.ipopt_button = QPushButton('Recover with IPOPT')
        self.ipopt_button.setVisible(False)
        self.ipopt_button.clicked.connect(
            lambda _checked=False: self.recover_with_ipopt()
        )
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.ipopt_button)
        left_layout.addLayout(status_row)
        top_row.addStretch()
        # Operating conditions, each change schedules a solve
        self.mass_flow_spin = _make_spin(
            self.backend.get_value(n0.oth.TotMassFlow), 0.0, 1e4, 0.5, 3
        )
        self.mass_flow_spin.valueChanged.connect(self._schedule_solve)
        top_row.addWidget(QLabel('Mass flow [kg/s]'))
        top_row.addWidget(self.mass_flow_spin)
        reset_button = QPushButton('Reset to last converged')
        reset_button.clicked.connect(lambda _checked=False: self.reset_geometry())
        top_row.addWidget(reset_button)
        initial_button = QPushButton('Reset to initial state')
        initial_button.clicked.connect(lambda _checked=False: self.reset_to_initial())
        top_row.addWidget(initial_button)
        export_button = QPushButton('Export script (Ctrl+E)')
        export_button.clicked.connect(lambda _checked=False: self.export_setup())
        top_row.addWidget(export_button)
        import_button = QPushButton('Import setup (Ctrl+I)')
        import_button.clicked.connect(lambda _checked=False: self.import_setup())
        top_row.addWidget(import_button)
        layout.addLayout(top_row)
        views_splitter = QSplitter(Qt.Orientation.Vertical)
        views_splitter.setChildrenCollapsible(False)
        views_splitter.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        views_splitter.addWidget(self.profile_view)
        views_splitter.addWidget(self.parabola_view)
        views_splitter.setStretchFactor(0, PROFILE_VIEW_STRETCH)
        views_splitter.setStretchFactor(1, PARABOLA_VIEW_STRETCH)
        layout.addWidget(views_splitter, 1)
        # The views are independent: each has its own scale, position and x range
        self.group = [self.profile_view, self.parabola_view]
        for view in self.group:
            # Ctrl + scroll zooms only the view under the cursor
            view.zoom_callback = lambda factor, view=view: self.zoom_views(
                factor, [view]
            )

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
        # Point positions of every row at the last converged solution
        self._converged_points: list[list[QPointF]] = []
        # Imposed flow angles of every row at the last converged solution
        self._converged_angles: list[dict[VarSpec, float]] = []
        # Inlet conditions and fluid at the last converged solution
        self._converged_inlet: dict[VarSpec, float] = {}
        self._converged_fluid: FluidChoice = self.backend.fluid

        # A loaded setup has several rows, each on the shaft that turns at its speed
        self._syncing = True  # setting the shaft speeds must not schedule a solve
        for index in range(self.backend.num_rows):
            omega = self.backend.get_value(self.backend.row_nodes(index)[1].kin.Omega)
            self._append_row(self._shaft_for_omega(omega))
        self._syncing = False
        self._fit_triangle_area()
        extra_columns = min(self.backend.num_rows, MAX_VISIBLE_ROWS) - 1
        self.resize(self.width() + extra_columns * TRIANGLE_COLUMN_WIDTH, self.height())
        self._update_triangles()
        self._remember_converged()
        # What the window showed at startup, for the reset button
        self._initial_state = (
            self._converged_points,
            self._converged_angles,
            self._converged_inlet,
            self._converged_fluid,
        )

        # Animation runs on the log of the zoom so steps compose smoothly
        self._zoom_applied = 0.0
        self._zoom_target = 0.0
        self._zoom_views: list[SyncedView] = self.group
        self._zoom_anim = QVariantAnimation(self)
        self._zoom_anim.setDuration(ZOOM_DURATION_MS)
        self._zoom_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._zoom_anim.valueChanged.connect(self._on_zoom_step)

        # Gliding fit of the views when a row is added
        self._fit_from: list[tuple[float, float, float]] = []
        self._fit_to: list[tuple[float, float, float]] = []
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
        QShortcut(QKeySequence('Ctrl+E'), self).activated.connect(
            lambda: self.export_setup()
        )
        QShortcut(QKeySequence('Ctrl+I'), self).activated.connect(
            lambda: self.import_setup()
        )

    def export_setup(self):
        """Save the setup as a Python script, an equivalent TOML file and the solution."""
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export setup', 'adet_setup.py', 'Python script (*.py)'
        )
        if not path:
            return
        # The backend holds what is drawn, converged or not
        self._solve_timer.stop()
        self.backend.set_geometry(self.drawn_geometry() | self.operating_conditions())
        # The files go next to each other in the chosen directory
        script = Path(path)
        toml = script.with_suffix('.toml')
        solution = script.with_name(f'{script.stem}_solution.pkl')
        try:
            self.backend.save_solution(solution)
            script.write_text(
                self.backend.export_script(solution.name), encoding='utf-8'
            )
            toml.write_text(self.backend.to_toml(solution.name), encoding='utf-8')
        except (OSError, ValueError) as err:
            logger.warning(f'Could not export the setup: {err}')
            self.status_label.setText('Could not export the setup')
            return
        self.status_label.setText(
            f'Exported {script.name}, {toml.name} and {solution.name} to {script.parent.name}/'
        )

    def import_setup(self):
        """Replace the window by one that shows a setup read from a TOML file."""
        path, _ = QFileDialog.getOpenFileName(
            self, 'Import setup', '', 'TOML setup (*.toml)'
        )
        if not path:
            return
        toml = Path(path)
        self._solve_timer.stop()
        try:
            backend = RowBackend.from_toml(
                toml.read_text(encoding='utf-8'), toml.parent
            )
        except Exception as err:  # bad file, unknown spec or a solve that fails
            logger.warning(f'Could not import the setup: {err}')
            self.status_label.setText(f'Could not import {toml.name}')
            return
        # The rows, shafts and views are built from the backend, so start a new window
        window = MainGuiView(backend)
        window.setGeometry(self.geometry())
        window.status_label.setText(f'Imported {toml.name}')
        window.show()
        # Kept alive by the application through this reference
        MainGuiView._windows = [window]
        self.close()

    def _shaft_for_omega(self, omega: float) -> ShaftView:
        """Shaft turning at ``omega``; a moving shaft is added if none does."""
        if omega == 0.0:
            return self.casing_shaft
        # The unused rotating shaft takes the first speed that is loaded
        if not self.rows and not math.isclose(
            self.rotor_shaft.omega, omega, abs_tol=0.05
        ):
            self.rotor_shaft.omega_spin.setValue(omega)
        for shaft in self.shaft_panel.shafts:
            # The spin boxes round the speed
            if math.isclose(shaft.omega, omega, abs_tol=0.05):
                return shaft
        shaft = self.shaft_panel.add_moving_shaft()
        shaft.omega_spin.setValue(omega)
        return shaft

    def _append_row(self, shaft: ShaftView | None = None):
        """Draw the last row of the backend after the rows already shown.

        The row starts on ``shaft``, by default the stationary casing.
        """
        row = BladeRowView(
            self.backend,
            len(self.rows),
            self.rows[-1] if self.rows else None,
            self.profile_scene,
            self.parabola_scene,
            self.shaft_panel.shafts,
            shaft or self.casing_shaft,
            self._schedule_solve,
            self._on_angle_mode_changed,
        )
        self.rows.append(row)
        # The radius axis of the first row reaches the highest point of every row
        if (axis := self.rows[0].radius_axis) is not None:
            axis.watch(row.points)
        row.shaft_combo.currentIndexChanged.connect(self._schedule_solve)

        # Shaft choice on top of the triangles of the row; all triangles share one scale
        column = QVBoxLayout()
        column.addWidget(row.row_label)
        column.addWidget(row.shaft_combo)
        column.addWidget(row.coefficients_label)
        for switch, triangle_view in zip(row.angle_switches, row.triangle_views):
            # Columns keep their width, so extra rows scroll instead of squeezing
            triangle_view.setMinimumWidth(TRIANGLE_COLUMN_WIDTH - ROW_GAP)
            column.addWidget(triangle_view.title_label)
            column.addWidget(switch)
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
            # The new row starts on the casing, so it must not inherit the last speed
            params = self.backend.next_row_params(self.casing_shaft.omega)
            self.backend.add_row(params)
        except RuntimeError as err:
            logger.warning(f'Could not add a row: {err}')
            self.status_label.setText('Could not solve with the added row')
            return
        self._append_row()
        self._update_triangles()
        self._fit_triangle_area()
        if len(self.rows) <= MAX_VISIBLE_ROWS:
            self.resize(self.width() + TRIANGLE_COLUMN_WIDTH, self.height())
        QApplication.processEvents()
        self.profile_view._update_extent()
        self._remember_converged()
        self.status_label.setText(f'Row {len(self.rows)} added and converged')

    def delete_row(self):
        """Remove the last row. The first row always stays."""
        if len(self.rows) < 2:
            return
        self._solve_timer.stop()
        # The remaining rows keep what is drawn, so it goes to the backend first
        values = self._inlet_conditions()
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
        for widget in (
            row.row_label,
            row.shaft_combo,
            row.coefficients_label,
            *(view.title_label for view in row.triangle_views),
            *row.angle_switches,
            *row.triangle_views,
        ):
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
        self._remember_converged()
        self.status_label.setText(f'Row {len(self.rows) + 1} removed and converged')

    def drawn_geometry(self) -> dict[VarSpec, float]:
        """Geometry currently drawn, as boundary conditions of the rows (m, rad)."""
        geometry: dict[VarSpec, float] = {}
        for row in self.rows:
            geometry |= row.drawn_geometry()
        return geometry

    def _inlet_conditions(self) -> dict[VarSpec, float]:
        """Mass flow (kg/s), total pressure (Pa) and temperature (K) of the inlet."""
        return {
            n0.oth.TotMassFlow: self.mass_flow_spin.value(),
            n0.tot.Pressure: self.inlet_panel.pressure,
            n0.tot.Temperature: self.inlet_panel.temperature,
        }

    def operating_conditions(self) -> dict[VarSpec, float]:
        """Inlet conditions and rotational speeds (rad/s) from the input fields."""
        conditions = self._inlet_conditions()
        for row in self.rows:
            conditions |= row.operating_conditions()
        return conditions

    def _on_fluid_changed(self, fluid: FluidChoice):
        """Rebuild the solution with another working fluid."""
        self._solve_timer.stop()
        # The current drawing is solved first, so the new fluid starts from it
        if not self.update_solution():
            self.status_label.setText('Fix the current solution before the fluid')
            self.fluid_panel.set_fluid(self.backend.fluid)
            return
        try:
            self.backend.set_fluid(fluid)
        except Exception as err:  # CoolProp raises ValueError for unknown fluids
            logger.warning(f'Could not change the fluid: {err}')
            self.status_label.setText('Could not solve with the other fluid')
            self.fluid_panel.set_fluid(self.backend.fluid)
            return
        self._update_triangles()
        self._remember_converged()
        self.status_label.setText('Fluid changed and converged')

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
        self.ipopt_button.setVisible(not converged)
        self._update_triangles(converged)
        if converged:
            self._sync_camber_lines()
            self._remember_converged()
        return converged

    def recover_with_ipopt(self):
        """Solve the drawn geometry with Ipopt after Newton failed (manual only)."""
        self._solve_timer.stop()
        self.backend.set_geometry(self.drawn_geometry() | self.operating_conditions())
        converged = self.backend.solve_ipopt()
        self._update_triangles(converged)
        if not converged:
            self.status_label.setText('IPOPT failed')
            return
        self.ipopt_button.setVisible(False)
        self._sync_camber_lines()
        self._remember_converged()
        self.status_label.setText('IPOPT converged')

    def _on_angle_mode_changed(self, row: BladeRowView, node: int, absolute: bool):
        """Make a row impose the absolute or relative flow angle at a node."""
        self._solve_timer.stop()
        # The current drawing is solved first, so the new boundary conditions are
        # the flow angles that are on screen
        if not self.update_solution():
            self.status_label.setText('Fix the current solution before switching')
            row.set_angle_mode()  # back to the switch of the backend
            return
        try:
            self.backend.set_angle_mode(row.index, node, absolute)
        except RuntimeError as err:
            logger.warning(f'Could not switch the flow angle: {err}')
            self.status_label.setText('Could not solve with the other flow angle')
            row.set_angle_mode()
            return
        for other in self.rows:
            other.set_angle_mode()
        self._update_triangles()
        self._remember_converged()
        self.status_label.setText(
            f'Row {row.index + 1} {"inlet" if node == 0 else "outlet"} imposes the '
            f'{"absolute" if absolute else "relative"} flow angle'
        )

    def _remember_converged(self):
        """Keep the drawn point positions and flow angles to go back to."""
        self._converged_points = [
            [point.pos() for point in row.points] for row in self.rows
        ]
        self._converged_angles = [dict(row.flow_angles) for row in self.rows]
        self._converged_inlet = self._inlet_conditions()
        self._converged_fluid = self.backend.fluid

    def reset_geometry(self):
        """Redraw the geometry of the last converged solution and solve it again.

        The inlet conditions and the fluid go back as well.
        """
        if len(self._converged_points) != len(self.rows):
            return
        for row, angles in zip(self.rows, self._converged_angles):
            # Only the angles of the kind the backend imposes now, as the other kind
            # (switched since the snapshot) is not a boundary condition any more
            row.flow_angles.update(
                {
                    spec: value
                    for spec, value in angles.items()
                    if spec in row.flow_angles
                }
            )
        self._solve_timer.stop()
        inlet = self._converged_inlet
        self.mass_flow_spin.blockSignals(True)
        self.mass_flow_spin.setValue(inlet[n0.oth.TotMassFlow])
        self.mass_flow_spin.blockSignals(False)
        self.inlet_panel.set_conditions(
            inlet[n0.tot.Pressure], inlet[n0.tot.Temperature]
        )
        self.fluid_panel.set_fluid(self._converged_fluid)
        self._syncing = True  # the restore is not an edit, so no solve per point
        try:
            # Points drag their aligned partners along, so a second pass settles
            # any point that a later one moved
            for _ in range(2):
                for row, positions in zip(self.rows, self._converged_points):
                    for point, position in zip(row.points, positions):
                        point.setPos(position)
        finally:
            self._syncing = False
        if self.update_solution():
            self.status_label.setText('Reset to the last converged geometry')

    def reset_to_initial(self):
        """Go back to the single row, geometry, inlet and fluid shown at startup."""
        while len(self.rows) > 1:
            count = len(self.rows)
            self.delete_row()
            if len(self.rows) == count:  # the backend could not drop the row
                return
        (
            self._converged_points,
            self._converged_angles,
            self._converged_inlet,
            self._converged_fluid,
        ) = self._initial_state
        self.reset_geometry()
        if self.status_label.text() == 'Reset to the last converged geometry':
            self.status_label.setText('Reset to the initial state')

    def _sync_camber_lines(self):
        """Draw the camber lines of the solved metal angles.

        Moving them is not an edit, so it must not trigger another solve. The rows go
        in order, as each trailing edge sets the height of the next leading edge.
        """
        self._syncing = True
        try:
            for row in self.rows:
                row.sync_camber()
        finally:
            self._syncing = False
        self.parabola_view._update_extent()

    def _update_triangles(self, converged: bool = True):
        """Show the velocity triangles of the current solution."""
        for row in self.rows:
            row.update_triangles(converged)
        if converged:
            VelocityTriangleView.fit_group(self._triangle_views())

    def zoom_views(self, factor: float, views: list[SyncedView] | None = None):
        """Animate a zoom of ``views`` (both by default) by the same factor.

        Presses during a running animation add to what is left of it, when they zoom
        the same views.
        """
        views = self.group if views is None else views
        remaining = 0.0
        if (
            self._zoom_anim.state() == QVariantAnimation.State.Running
            and views == self._zoom_views
        ):
            remaining = self._zoom_target - self._zoom_applied
        self._zoom_anim.stop()
        self._zoom_views = views
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
        """Zoom the views being zoomed by the same factor, each about its own centre."""
        for view in self._zoom_views:
            viewport = view.viewport()
            assert viewport is not None
            center = view.mapToScene(viewport.rect().center())
            view.scale(factor, factor)
            view.centerOn(center)

    def _fit_target(
        self, horizontal_only: bool = False
    ) -> list[tuple[float, float, float]]:
        """Scale and centre (x, y) of every view that fit its own content.

        With ``horizontal_only`` the scale only makes the content fit the width.
        """
        for view in self.group:
            view._update_extent()
        targets = []
        for view in self.group:
            rect = view.content_rect()
            viewport = view.viewport()
            assert viewport is not None
            scale = viewport.width() / rect.width()
            if not horizontal_only:
                scale = min(scale, viewport.height() / rect.height())
            targets.append((scale, rect.center().x(), rect.center().y()))
        return targets

    def _show_view(self, targets: list[tuple[float, float, float]]):
        for view, (scale, center_x, center_y) in zip(self.group, targets):
            view.resetTransform()
            view.scale(scale, scale)
            view.centerOn(center_x, center_y)

    def _current_view(self) -> list[tuple[float, float, float]]:
        """Scale and centre (x, y) of every view as shown now."""
        current = []
        for view in self.group:
            viewport = view.viewport()
            assert viewport is not None
            center = view.mapToScene(viewport.rect().center())
            current.append((view.transform().m11(), center.x(), center.y()))
        return current

    def fit_views(self, animated: bool = True, horizontal_only: bool = False):
        """Fit the profile and the parabola (not the axis), each to its own content.

        With ``animated`` (the default) the views glide to the fit with an ease.
        With ``horizontal_only`` the fit only considers the width of the content.
        """
        self._zoom_anim.stop()
        self._fit_anim.stop()
        end = self._fit_target(horizontal_only)
        if not animated:
            self._show_view(end)
            return
        self._fit_from, self._fit_to = self._current_view(), end
        self._fit_anim.start()

    def _on_fit_step(self, progress):
        """Show the views a fraction ``progress`` of the way to the fit."""
        t = float(progress)
        steps = []
        for (scale0, x0, y0), (scale1, x1, y1) in zip(self._fit_from, self._fit_to):
            # The scale is interpolated in its log, like the zoom, so it feels uniform
            scale = math.exp(
                math.log(scale0) + (math.log(scale1) - math.log(scale0)) * t
            )
            steps.append((scale, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        self._show_view(steps)
