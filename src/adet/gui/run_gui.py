"""Entry point of the GUI: splash screen while the first solution is computed."""

import threading
import time
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PyQt6.QtWidgets import QApplication, QSplashScreen

from adet.gui.row_backend import RowBackend
from adet.gui.views import MainGuiView
from adet.tools.printing import logo_morph_frame

SPLASH_SIZE = (640, 300)  # pixels
SPLASH_FONT_FAMILIES = ('Cascadia Mono', 'Cascadia Code', 'Consolas')  # terminal fonts
SPLASH_FONT_SIZE = 16  # pixels
SPLASH_TEXT_COLOR = QColor(200, 150, 220)  # purple, like the terminal logo
SPLASH_DURATION = 1.5  # seconds
SPLASH_FPS = 30.0
SPLASH_BACKGROUND = QColor(40, 40, 40)  # same as the scene background
SPLASH_RADIUS = 24  # corner radius in pixels
FONTS_DIR = (
    Path(__file__).resolve().parents[3] / 'fonts'
)  # repository root fonts folder


def _splash_pixmap(lines: list[str]) -> QPixmap:
    """Splash image with the ASCII logo ``lines`` on a dark rounded background."""
    width, height = SPLASH_SIZE
    canvas = QPixmap(width, height)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    background = QPainterPath()
    background.addRoundedRect(QRectF(0, 0, width, height), SPLASH_RADIUS, SPLASH_RADIUS)
    painter.fillPath(background, SPLASH_BACKGROUND)
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    # Prefer the fonts of the Windows terminals, then the system fixed font
    font.setFamilies([*SPLASH_FONT_FAMILIES, font.family()])
    font.setPixelSize(SPLASH_FONT_SIZE)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(SPLASH_TEXT_COLOR)
    # Draw on a fixed character grid, so the columns never drift between lines
    metrics = QFontMetricsF(font)
    cell_width = metrics.horizontalAdvance('M')
    cell_height = metrics.lineSpacing()
    left = (width - cell_width * max(len(line) for line in lines)) / 2
    top = (height - cell_height * len(lines)) / 2 + metrics.ascent()
    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            if not char.isspace():
                painter.drawText(
                    QPointF(left + col * cell_width, top + row * cell_height), char
                )
    painter.end()
    return canvas


def make_splash() -> QSplashScreen:
    """Splash screen showing the start of the logo animation."""
    splash = QSplashScreen(_splash_pixmap(logo_morph_frame(0.0)))
    # Let the transparent corners show through
    splash.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    return splash


def animate_splash(app: QApplication, splash: QSplashScreen, worker: threading.Thread):
    """Play the logo morph on the splash screen until ``worker`` has finished.

    The morph always plays completely and the final logo stays up while waiting.
    """
    start = time.perf_counter()
    while (progress := (time.perf_counter() - start) / SPLASH_DURATION) < 1:
        splash.setPixmap(_splash_pixmap(logo_morph_frame(progress)))
        app.processEvents()
        time.sleep(1 / SPLASH_FPS)
    splash.setPixmap(_splash_pixmap(logo_morph_frame(1.0)))
    while worker.is_alive():
        app.processEvents()
        worker.join(1 / SPLASH_FPS)


def load_app_font(app: QApplication):
    """Register JetBrains Mono from the repository fonts folder and make it the app font."""
    for font_file in FONTS_DIR.glob('JetBrainsMono*.ttf'):
        QFontDatabase.addApplicationFont(str(font_file))
    if 'JetBrains Mono' in QFontDatabase.families():
        app.setFont(QFont('JetBrains Mono', 10))


def main():
    """Run the test application."""
    app = QApplication([])
    load_app_font(app)
    # The first solution is computed in a thread while the splash animation plays
    backends: list[RowBackend] = []
    errors: list[BaseException] = []

    def solve_first():
        try:
            backends.append(RowBackend())
        except BaseException as error:
            errors.append(error)

    solver = threading.Thread(target=solve_first, daemon=True)
    solver.start()
    splash = make_splash()
    splash.show()
    animate_splash(app, splash, solver)
    if errors:
        raise errors[0]
    view = MainGuiView(backends[0])
    view.show()
    splash.finish(view)
    app.exec()


if __name__ == '__main__':
    print('Starting the GUI...')
    main()
