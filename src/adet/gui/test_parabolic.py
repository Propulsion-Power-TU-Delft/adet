"""Test script for ParabolicLine class."""

import sys

from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from adet.gui.entities import DraggablePoint, ParabolicLine


def main():
    app = QApplication(sys.argv)

    scene = QGraphicsScene()
    scene.setSceneRect(0, 0, 800, 600)

    # Create start, control, and end points
    start = DraggablePoint(100, 300, radius=6)
    control = DraggablePoint(400, 100, radius=6)
    end = DraggablePoint(700, 300, radius=6)

    scene.addItem(start)
    scene.addItem(control)
    scene.addItem(end)

    # Create parabolic line
    parabolic = ParabolicLine(start, control, end)
    scene.addItem(parabolic)

    # Create view
    view = QGraphicsView(scene)
    view.setRenderHint(QPainter.RenderHint.Antialiasing)
    view.setWindowTitle('Parabolic Line with Vertical Constraints')
    view.resize(800, 600)

    view.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
