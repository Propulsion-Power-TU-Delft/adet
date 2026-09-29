"""Test view for draggable points and connecting lines."""

from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView
from PyQt6.QtGui import QPainter
from adet.gui.entities import DraggablePoint, Line, SymmetricLine, CubicBezierLine


class EntitiesTestView(QGraphicsView):
    """A simple view to test draggable points and lines."""

    def __init__(self):
        scene = QGraphicsScene()
        super().__init__(scene)

        self.setWindowTitle('Draggable Points and Lines Test')
        self.setGeometry(100, 100, 800, 600)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create draggable points with different constraints
        # Blue: free movement
        self.point_free = DraggablePoint(100, 100)
        # Red: x-axis constrained (horizontal only)
        self.point_x_constrained = DraggablePoint(300, 200, axis_constraint='x')
        # Green: y-axis constrained (vertical only)
        self.point_y_constrained = DraggablePoint(200, 400, axis_constraint='y')

        # Add points to scene
        scene.addItem(self.point_free)
        scene.addItem(self.point_x_constrained)
        scene.addItem(self.point_y_constrained)

        # Create lines connecting points
        self.line1 = Line(self.point_free, self.point_x_constrained)
        self.line2 = Line(self.point_x_constrained, self.point_y_constrained)
        self.line3 = Line(self.point_y_constrained, self.point_free)

        # Add lines to scene
        scene.addItem(self.line1)
        scene.addItem(self.line2)
        scene.addItem(self.line3)

        # Create a symmetric line (center point controls translation,
        # endpoints are always symmetric about center)
        self.symmetric_center = DraggablePoint(600, 150)
        self.symmetric_end = DraggablePoint(700, 150)
        self.symmetric_line = SymmetricLine(self.symmetric_center, self.symmetric_end)

        # Add symmetric line points and line to scene
        scene.addItem(self.symmetric_center)
        scene.addItem(self.symmetric_end)
        scene.addItem(self.symmetric_line.end2)
        scene.addItem(self.symmetric_line)

        # Create cubic Bezier curve
        self.bezier_start = DraggablePoint(50, 500)
        self.bezier_control1 = DraggablePoint(150, 300)
        self.bezier_control2 = DraggablePoint(550, 250)
        self.bezier_end = DraggablePoint(750, 500)
        self.cubic_bezier = CubicBezierLine(
            self.bezier_start,
            self.bezier_control1,
            self.bezier_control2,
            self.bezier_end,
        )

        # Add cubic Bezier points and line to scene
        scene.addItem(self.bezier_start)
        scene.addItem(self.bezier_control1)
        scene.addItem(self.bezier_control2)
        scene.addItem(self.bezier_end)
        scene.addItem(self.cubic_bezier)

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
