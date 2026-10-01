import random
import sys
import time
from datetime import datetime
from pathlib import Path

from art import tprint

current_year = datetime.now().year

FOOTER = """
| >>> | Developed and maintained by Francesco Vaccari
| >>> | Propulsion & Power, Faculty of Aerospace Engineering, TU Delft
| >>> | 2024-2026
"""


def print_header():
    tprint('ADeT', font='isometric1')
    print(FOOTER)


def print_logo():
    logo_file = Path(__file__).parent / 'ascii_logo.txt'
    with open(logo_file) as file:
        logo = file.read()
    print('\033[92m' + logo + '\033[0m')
    print(FOOTER)


def logo_morph_frame(progress: float, band: int = 12) -> list[str]:
    """Lines of the logo at ``progress`` (0 to 1) of the morph from ``-`` to the logo.

    Every non-blank character starts as ``-``. A wave front sweeps from the left to
    the right: characters ahead of it are still ``-``, characters inside the ``band``
    columns behind the front flicker through random logo characters, and characters
    further behind are locked into their final value.
    """
    logo_file = Path(__file__).parent / 'ascii_logo.txt'
    lines = logo_file.read_text().splitlines()
    alphabet = sorted({char for line in lines for char in line if not char.isspace()})
    width = max(len(line) for line in lines)
    front = progress * (width + band)  # column where characters start morphing
    rows = []
    for line in lines:
        row = list(line)
        for col, char in enumerate(row):
            if char.isspace() or col < front - band:
                continue  # blank or already settled
            row[col] = random.choice(alphabet) if col < front else '-'
        rows.append(''.join(row))
    return rows


def logo_morph_bold(progress: float, band: int = 12) -> list[list[bool]]:
    """Per character of the logo, whether it is bold at ``progress`` of the morph.

    Characters are normal weight until the wave front reaches them and bold after.
    """
    logo_file = Path(__file__).parent / 'ascii_logo.txt'
    lines = logo_file.read_text().splitlines()
    width = max(len(line) for line in lines)
    front = progress * (width + band)
    return [[col < front for col in range(len(line))] for line in lines]


def animate_logo(duration: float = 1.5, fps: float = 30.0, band: int = 12):
    """Print the logo morphing from ``-``
    characters to the final logo, left to right."""
    lines = logo_morph_frame(1.0, band)  # the final logo

    def frame(progress: float) -> str:
        return '\n'.join(logo_morph_frame(progress, band))

    def draw(text: str, first: bool):
        if not first:
            sys.stdout.write(f'\033[{len(lines)}F')  # back to the first line
        sys.stdout.write('\033[92m' + text + '\033[0m\n')
        sys.stdout.flush()

    sys.stdout.write('\033[?25l')  # hide the cursor
    try:
        start = time.perf_counter()
        first = True
        while (progress := (time.perf_counter() - start) / duration) < 1:
            draw(frame(progress), first)
            first = False
            time.sleep(1 / fps)
        draw('\n'.join(lines), first)
    finally:
        sys.stdout.write('\033[?25h')  # show the cursor
    print(FOOTER)
