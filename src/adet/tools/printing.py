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
