from collections import defaultdict, namedtuple
import csv
from pathlib import Path

import numpy as np
import jax
from pint import Quantity

from adet.components import BladeRow, ComponentNetwork, Inlet
from adet.components.connections import Shaft
from adet.assembly import CasadiSystem, solve_problem
from adet.equations.geometrical import MinimalCamberLine
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.fluid.settings import FluidSettings, ExternalFluidModel
from adet.registries import DefaultUnitsRegistry, GuessRegistry, ScalingRegistry
from adet.tools.coolprop_utils import DebugAbstractState


# Import Data
'../../../data/opencases/tfd_4ac/'

data_folder = Path(__file__).parents[3] / 'data/opencases/tfd_4ac/'

ROWS = ['IGV', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3', 'R4', 'S4']


def extract_tsv_data(file_path, as_columns=False):
    """Extract data from TSV file and return namedtuple or list of namedtuples.

    Args:
        file_path: Path to TSV file
        as_columns: If True, return single namedtuple with lists as fields.
                   If False, return list of namedtuples (one per row).
    """
    with open(file_path) as file:
        lines = file.readlines()
        header = lines[0]
        fields = header.replace('\n', '').split('\t')
        data = lines[1:]
        csv_data = csv.reader(data, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)

        RowData = namedtuple('RowData', fields)

        if as_columns:
            # Return single namedtuple with each field as a list
            columns = {field: [] for field in fields}
            for entry in csv_data:
                for idx, field in enumerate(fields):
                    columns[field].append(entry[idx])
            return RowData(**columns)
        else:
            # Return list of namedtuples
            result = []
            for entry in csv_data:
                result.append(RowData(*entry))
            return result


# Data extraction
spanwise_data = {}
for row in ROWS:
    spanwise_data[row] = extract_tsv_data(
        data_folder / f'{row}_spanwise_cold.tsv', as_columns=True
    )

integral_data = {}
for data_entry in extract_tsv_data(
    data_folder / 'integral_geometry_cold.tsv', as_columns=False
):
    # Use first field as key (row identifier)
    integral_data[data_entry[0]] = data_entry


def interpolate_spanwise_data(
    spanwise_data, mean_radius: float, height: float, num_stations: int
):
    """Find the closest spanwise stations based on mean radius and channel height.

    Args:
        mean_radius: Mean radius at inlet from integral data (rmid_inlet)
        height: Channel height at inlet from integral data (height_inlet)
        spanwise_data_tuple: Named tuple with spanwise data columns
        num_stations: Number of spanwise stations to return (default: 5)

    Returns:
        List of indices for the closest spanwise stations, sorted from hub to tip.
        If only one station exists in spanwise data, returns [0] regardless
        of num_stations.
    """
    # Get the radii from spanwise data
    radii = np.array(spanwise_data.radius_inlet)

    # Cell-centered radius definitions
    hub_radius = mean_radius - height / 2.0 + height / num_stations / 2
    tip_radius = mean_radius + height / 2.0 - height / num_stations / 2

    # Target radii: evenly spaced from hub to tip
    target_radii = np.linspace(hub_radius, tip_radius, num_stations)

    if num_stations == 1:
        target_radii = np.array([mean_radius])

    # Interpolate data from closest radius
    interp_data = {}
    for field in spanwise_data._fields:
        interp_values = np.interp(
            target_radii,
            radii,
            getattr(spanwise_data, field),
        )
        interp_data[field] = interp_values

    return interp_data


# Define blade rows using imported data
# Each BladeRow needs inlet/outlet constraints matching the spanwise and integral data


def create_blade_row(row_name: str, shaft_connection: Shaft, spanwise_stations: int):
    """Create a BladeRow using integral and spanwise data.

    Args:
        row_name: Name of the blade row (e.g., 'IGV', 'R1', 'S1')
        shaft_connection: Shaft object (casing for stators, shaft for rotors)
        spanwise_stations: Number of spanwise stations to use

    Returns:
        BladeRow object with constraints from data
    """
    # Get data for this row
    integral = integral_data[row_name]
    spanwise = spanwise_data[row_name]

    rmid_inlet = integral_data[row_name].rmid_inlet
    height_inlet = integral_data[row_name].rmid_inlet

    rmid_outlet = integral_data[row_name].rmid_outlet
    height_outlet = integral_data[row_name].rmid_outlet

    inlet_data = interpolate_spanwise_data(
        spanwise,
        rmid_inlet,
        height_inlet,
        spanwise_stations,
    )

    outlet_data = interpolate_spanwise_data(
        spanwise,
        rmid_outlet,
        height_outlet,
        spanwise_stations,
    )

    # WARN: Missing axial chord

    # Create blade row with constraints
    return BladeRow(
        row_name,
        shaft=shaft_connection,
        in_constraints={
            'geo': {
                'rmid': rmid_inlet,
                'height': height_inlet,
                'meridional_angle': 0.0,
                'bld_thick': inlet_data['bld_thick_inlet'],
                'metal_angle': inlet_data['metal_angle_inlet'],
            },
        },
        out_constraints={
            'geo': {
                'rmid': rmid_outlet,
                'height': height_outlet,
                'meridional_angle': 0.0,
                'bld_thick': outlet_data['bld_thick_outlet'],
                'metal_angle': outlet_data['metal_angle_outlet'],
                'chord_ax': outlet_data['chord_axial'],
                'num_blades': integral.num_blades,
            },
        },
        extra_equations={
            # Camberline model (minimal for now)
            MinimalCamberLine(): (0, 1),
            # Follow the blade geometry
            ZeroDeviation(): 1,
            # Zero loss for now (isentropic)
            PercentageEntropyLoss(0.0): (0, 1),
        },
    )


# === CONFIGURATION ===
NUM_SPAN = 3
SCALED = True
PLOTS = True
PRINTS = True

# === FLUID MODEL SETUP ===
# Real gas model using CoolProp
abs_state = DebugAbstractState('HEOS', 'Air')
abs_state.debug_print = False
real_model = ExternalFluidModel(abs_state)

_dfu_reg = DefaultUnitsRegistry()
_scl_reg = ScalingRegistry()
_gss_reg = GuessRegistry()

# Configure fluid settings
settings = FluidSettings(
    model=real_model,
    update_variables=('p', 'T', 'rhomass', 'smass', 'hmass'),
    update_length=2,
)

# Add units for custom variables
_dfu_reg.from_dict(
    {
        'delta_smass_pct': 'J/ (kg * K)',
        'percentage_loss': 'dimensionless',
        'workCoeff': 'dimensionless',
        'flowCoeff': 'dimensionless',
        'specificSpeed': 'dimensionless',
        'STratio': 'dimensionless',
        'VmRatio': 'dimensionless',
        'sizeParameter': 'meters',
        # Profile losses
        'Cd_profile': 'dimensionless',
        'xi_by_camb_len_A': 'meters',
        'xi_by_camb_len_B': 'meters',
        'k_prof': 'dimensionless',
        'mom_by_bld_thick': 'dimensionless',
        'disp_by_mom_thick': 'dimensionless',
    }
)

# Set fallback values for scales and guesses if needed
# _scl_reg.set_fallback_value(1.0)
_gss_reg.set_fallback_value(1.0)

# === SHAFT DEFINITIONS ===
casing = Shaft(omega=0.0, is_constrained=True)
shaft = Shaft(omega=Quantity(14400, 'rpm'), is_constrained=True)

# === INLET CONDITIONS ===
# Get inlet data from IGV inlet
igv_integral = integral_data['IGV']
inlet = Inlet(
    {
        'kin': {
            'alpha': Quantity(0, 'deg'),  # Axial inlet
            'Vm': Quantity(80, 'm/s'),  # Assumed inlet velocity
        },
        'tot': {
            'p': 1.013e5,  # Assumed inlet pressure (atmospheric)
            'T': 288.15,  # Assumed inlet temperature (15°C)
        },
    }
)

igv = create_blade_row('IGV', casing, NUM_SPAN)
r1 = create_blade_row('R1', shaft, NUM_SPAN)
s1 = create_blade_row('S1', casing, NUM_SPAN)
r2 = create_blade_row('R2', shaft, NUM_SPAN)
s2 = create_blade_row('S2', casing, NUM_SPAN)
r3 = create_blade_row('R3', shaft, NUM_SPAN)
s3 = create_blade_row('S3', casing, NUM_SPAN)
r4 = create_blade_row('R4', shaft, NUM_SPAN)
s4 = create_blade_row('S4', casing, NUM_SPAN)

# === CREATE NETWORK ===
ntw = ComponentNetwork(
    settings,  # Fluid settings
    inlet,  # Inlet conditions
    CasadiSystem(spanwise_stations=NUM_SPAN),  # Backend
    igv,
    # r1,
    # s1,
    # r2,
    # s2,
    # r3,
    # s3,
    # r4,
    # s4,
)
# === GLOBAL CONSTRAINTS ===
ntw.system.add_global_constraints(
    {
        'oth': {
            # Ideal gas properties (reference)
            'disp_thick': 0.0,  # Ignore boundary layer blockage
            # Profile loss coefficients
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,
            'xi_by_camb_len_B': 0.675,
        }
    }
)

# === BUILD AND SOLVE ===
ntw.system.build(SCALED)

rootfinder = ntw.system.make_rootfinder('nlpsol')
x0 = ntw.system.get_initial_guess()
kn = ntw.system.get_scaled_constraints()

sol = solve_problem(rootfinder, x0, kn)

# Write solution to network
ntw.system.write_solution_to_nodes(sol)

# === POST-PROCESS ===
if PRINTS:
    for i, node in enumerate(ntw.system.nodes):
        to_print = f"""
###################
##### NODE {i:<2} #####
###################
{node}

"""
        print(to_print)

    ntw.print_structure()

if PLOTS:
    import matplotlib.pyplot as plt

    FONTSIZE = 18

    # Plot velocity triangles
    for i, n in enumerate(ntw.system.nodes):
        n.kin.plot(n.geo, FONTSIZE)
        plt.title(f'Node {i} - {ROWS[i // 2] if i < len(ROWS) * 2 else "Exit"}')

    plt.show()
