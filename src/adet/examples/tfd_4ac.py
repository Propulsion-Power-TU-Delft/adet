from collections import defaultdict
import csv
from pathlib import Path
from pint import Quantity

from adet.components import BladeRow
from adet.components.connections import Shaft


casing = Shaft(omega=0.0, is_constrained=True)
shaft = Shaft(omega=Quantity(14400, 'rpm'), is_constrained=True)

# Import Data
'../../../data/opencases/tfd_4ac/'

data_folder = Path(__file__).parents[3] / 'data/opencases/tfd_4ac/'

ROWS = ['IGV', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3', 'R4', 'S4']

# Initiate dict
rows_data = {row: defaultdict(list) for row in ROWS}
for row in ROWS:
    with open(data_folder / f'spanwise_geometry_{row}_cold.tsv') as file:
        lines = file.readlines()  # Skip header
        header = lines[0]
        fields = header.replace('\n', '').split('\t')
        data = lines[1:]
        geom_data = csv.reader(data, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)

        for entry in geom_data:
            for idx, field in enumerate(fields):
                rows_data[row][field].append(entry[idx])


# Define blade rows
# igv = BladeRow('igv')
