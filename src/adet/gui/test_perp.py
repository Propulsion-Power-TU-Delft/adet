"""Test view for perpendicular points constraint."""

from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from adet.gui.entities import (
    DraggablePoint,
    PerpendicularPoints,
)


class EntitiesTestView(QGraphicsView):
    """A simple view to test draggable points and lines."""

    def __init__(self):
        scene = QGraphicsScene()
        super().__init__(scene)

        self.setWindowTitle('Perpendicular Points Constraint Test')
        self.setGeometry(100, 100, 800, 600)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Test 1: Simple perpendicular constraint with 3 points
        # point1 (draggable) -> point2 (draggable)
        # -> point3 (calculated, fixed distance)
        # The angle at point2 is always 90 degrees, point3 is at constant distance
        self.point1 = DraggablePoint(150, 150, radius=5)
        self.point2 = DraggablePoint(300, 150, radius=5)

        # Create perpendicular constraint; point3 is calculated
        self.perp = PerpendicularPoints(self.point1, self.point2, point3_distance=50)

        # Add draggable points to scene
        scene.addItem(self.point1)
        scene.addItem(self.point2)

        # Add the perpendicular constraint visualization
        scene.addItem(self.perp)

        # Test 2: Another perpendicular constraint elsewhere
        self.point4 = DraggablePoint(500, 100, radius=5)
        self.point5 = DraggablePoint(600, 200, radius=5)

        self.perp2 = PerpendicularPoints(self.point4, self.point5, point3_distance=50)

        scene.addItem(self.point4)
        scene.addItem(self.point5)
        scene.addItem(self.perp2)

        # Set scene rect
        scene.setSceneRect(-50, -50, 900, 700)

    def mouseMoveEvent(self, event):
        """Update line geometry when mouse moves (for dragging)."""
        scene = self.scene()
        if scene:
            scene.update()
        super().mouseMoveEvent(event)


def main():
    """Run the test application."""
    app = QApplication([])
    view = EntitiesTestView()
    view.show()
    app.exec()


if __name__ == '__main__':
    main()
