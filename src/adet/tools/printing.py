from pathlib import Path
from art import tprint
from datetime import datetime

FOOTER = f"""
| ~ | Developed and maintained by Francesco Vaccari
| ~ | Propulsion & Power, Faculty of Aerospace Engineering, TU Delft
| ~ | 2024-{datetime.now().year}
"""


def print_header():
    tprint('ADeT', font='isometric1')
    print(FOOTER)


def print_logo():
    logo_file = Path(__file__).parents[3] / 'docs/logo/ascii_logo'
    with open(logo_file) as file:
        logo = file.read()
    print(logo)
    print(FOOTER)
