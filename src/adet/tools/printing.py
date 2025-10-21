from art import tprint
from datetime import datetime


def print_header():
    tprint('ADeT', font='isometric1')
    print(
        f'| ~ | Developed and maintained by Francesco Vaccari\n'
        f'| ~ | Propulsion & Power, Faculty of Aerospace Engineering, TU Delft\n'
        f'| ~ | 2024-{datetime.now().year}\n'
    )
