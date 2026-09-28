"""Interactive GUI for designing turbomachinery blade row profiles using PyQt."""

import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QGroupBox,
    QGridLayout,
    QPushButton,
)
from PyQt6.QtGui import QFont

from adet.components.blade_row import RowGeometry


@dataclass
class ProfileState:
    """Mutable state for a blade row profile."""

    r_in: float
    r_out: float
    height_in: float
    height_out: float
    mer_angle_in: float
    mer_angle_out: float
    axial_chord: float
    axial_offset: float = 0.0

    def create_geometry(self):
        """Create a RowGeometry object from current state."""
        return RowGeometry(
            r_in=self.r_in,
            r_out=self.r_out,
            height_in=self.height_in,
            height_out=self.height_out,
            mer_angle_in=self.mer_angle_in,
            mer_angle_out=self.mer_angle_out,
            axial_chord=self.axial_chord,
            axial_offset=self.axial_offset,
            force_straight=True,
        )


class ProfileEditorWindow(QMainWindow):
    """Main window for the profile editor."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Turbomachinery Profile Editor')
        self.setGeometry(100, 100, 1200, 700)

        # Create central widget and main layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Create control panel
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)

        # Title
        title = QLabel('Profile Editor')
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        control_layout.addWidget(title)

        # Profile parameters group
        params_group = QGroupBox('Profile Parameters')
        params_layout = QGridLayout()

        self.spinboxes = {}

        # Create spinbox controls
        parameters = [
            ('r_in', 'Inlet mid-radius (m)', 0.05, 0.5, 0.0001),
            ('r_out', 'Outlet mid-radius (m)', 0.05, 0.5, 0.0001),
            ('height_in', 'Inlet height (m)', 0.01, 0.2, 0.0001),
            ('height_out', 'Outlet height (m)', 0.01, 0.2, 0.0001),
            (
                'mer_angle_in',
                'Inlet meridional angle (rad)',
                -0.5,
                0.5,
                0.01,
            ),
            (
                'mer_angle_out',
                'Outlet meridional angle (rad)',
                -0.5,
                0.5,
                0.01,
            ),
            ('axial_chord', 'Axial chord (m)', 0.01, 0.5, 0.0001),
        ]

        row = 0
        for attr, label, min_val, max_val, step in parameters:
            params_layout.addWidget(QLabel(label), row, 0)

            spinbox = QDoubleSpinBox()
            spinbox.setMinimum(min_val)
            spinbox.setMaximum(max_val)
            spinbox.setSingleStep(step)
            spinbox.setDecimals(5)
            spinbox.valueChanged.connect(self.on_parameter_changed)
            params_layout.addWidget(spinbox, row, 1)

            self.spinboxes[attr] = spinbox
            row += 1

        params_group.setLayout(params_layout)
        control_layout.addWidget(params_group)

        # Info display
        info_group = QGroupBox('Current Geometry')
        info_layout = QVBoxLayout()
        self.info_label = QLabel()
        info_font = QFont('Courier')
        info_font.setPointSize(9)
        self.info_label.setFont(info_font)
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)

        # Button controls
        button_layout = QHBoxLayout()
        preview_btn = QPushButton('Preview Profile')
        preview_btn.clicked.connect(self.preview_profile)
        button_layout.addWidget(preview_btn)
        control_layout.addLayout(button_layout)

        control_layout.addStretch()

        layout.addLayout(control_layout)

        # Instructions panel
        instructions_group = QGroupBox('Instructions')
        instructions_layout = QVBoxLayout()
        instructions_text = QLabel(
            '1. Adjust parameters using the spinboxes on the left\n'
            '2. Spinboxes are synchronized with draggable points in preview\n'
            '3. Red circles: mid-radius (drag to move profile radially)\n'
            '4. Blue squares: height points (drag to adjust channel height)\n'
            "5. Click 'Preview Profile' to see the meridional geometry"
        )
        instructions_text.setWordWrap(True)
        instructions_layout.addWidget(instructions_text)
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)

        # Initialize state
        self.state = ProfileState(
            r_in=0.1,
            r_out=0.10,
            height_in=0.05,
            height_out=0.05,
            mer_angle_in=0.0,
            mer_angle_out=0.0,
            axial_chord=0.08,
            axial_offset=0.0,
        )

        self.update_spinboxes()
        self.update_info()

    def on_parameter_changed(self):
        """Update state when a spinbox value changes."""
        for attr, spinbox in self.spinboxes.items():
            setattr(self.state, attr, spinbox.value())

        self.update_info()

    def update_spinboxes(self):
        """Update spinbox values from state."""
        for attr, spinbox in self.spinboxes.items():
            spinbox.blockSignals(True)
            spinbox.setValue(getattr(self.state, attr))
            spinbox.blockSignals(False)

    def update_info(self):
        """Update the info display."""
        info_text = (
            f'Inlet radius:      {self.state.r_in:.5f} m\n'
            f'Outlet radius:     {self.state.r_out:.5f} m\n'
            f'Inlet height:      {self.state.height_in:.5f} m\n'
            f'Outlet height:     {self.state.height_out:.5f} m\n'
            f'Inlet angle:       {self.state.mer_angle_in:.5f} rad\n'
            f'Outlet angle:      {self.state.mer_angle_out:.5f} rad\n'
            f'Axial chord:       {self.state.axial_chord:.5f} m'
        )
        self.info_label.setText(info_text)

    def preview_profile(self):
        """Preview the profile in a matplotlib window."""
        geometry = self.state.create_geometry()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('Axial position (m)')
        ax.set_ylabel('Radius (m)')
        ax.set_title('Meridional Profile Preview')

        geometry.plot_meridional_profile(ax=ax, color='black')

        # Plot draggable points
        z_in = geometry.in_geo.z_offset
        r_in = self.state.r_in
        ax.plot(z_in, r_in, 'ro', markersize=10, label='Mid-radius (inlet)')

        z_out = geometry.out_geo.z_offset
        r_out = self.state.r_out
        ax.plot(z_out, r_out, 'ro', markersize=10, label='Mid-radius (outlet)')

        r_tip_in = geometry.in_geo.r1
        ax.plot(
            z_in,
            r_tip_in,
            'bs',
            markersize=10,
            label='Height tip (inlet)',
        )

        r_tip_out = geometry.out_geo.r1
        ax.plot(
            z_out,
            r_tip_out,
            'bs',
            markersize=10,
            label='Height tip (outlet)',
        )

        ax.legend()
        plt.tight_layout()
        plt.show()


def main():
    """Launch the profile editor application."""
    app = QApplication(sys.argv)
    window = ProfileEditorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
