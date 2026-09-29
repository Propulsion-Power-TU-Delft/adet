"""Test view for the meridional profile."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from adet.gui.entities import (
    AlignedPoints,
    DraggablePoint,
    MeridionalProfile,
    ParabolicLine,
)


class MeridionalProfileTestView(QGraphicsView):
    """A view showing a draggable meridional profile with its control polygons."""

    def __init__(self):
        scene = QGraphicsScene()
        super().__init__(scene)

        self.setWindowTitle('Meridional Profile Test')
        self.setGeometry(100, 100, 800, 600)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Default profile shifted 250 up so the whole drawing is centered in the view
        self.profile = MeridionalProfile(
            center1=(150, 350),
            end1=(150, 150),
            center2=(550, 250),
            end2=(550, 50),
        )
        self.profile.add_to_scene(scene)

        # Parabolic line below the profile, its endpoints sharing the x of the centers
        center1 = self.profile.center1
        center2 = self.profile.center2
        start = DraggablePoint(center1.pos().x(), 650)
        end = DraggablePoint(center2.pos().x(), 650)
        self.start_alignment = AlignedPoints(center1, start, 'y')
        self.end_alignment = AlignedPoints(center2, end, 'y')
        control = DraggablePoint((start.pos().x() + end.pos().x()) / 2, 700)
        self.parabola = ParabolicLine(start, control, end)
        for item in (start, end, control, self.parabola):
            scene.addItem(item)

        # Drawing spans x 150..550, y 50..700; the rect is centered around it
        scene.setSceneRect(-100, -50, 900, 800)

    def _fit_scene(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_scene()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_scene()

    def mouseMoveEvent(self, event):
        """Update the scene while dragging."""
        scene = self.scene()
        if scene:
            scene.update()
        super().mouseMoveEvent(event)


def main():
    """Run the test application."""
    app = QApplication([])
    view = MeridionalProfileTestView()
    view.show()
    app.exec()


if __name__ == '__main__':
    main()
