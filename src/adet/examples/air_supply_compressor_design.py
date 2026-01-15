# Design problem:
# Tt1 = 7.8 degC
# pt1 = 0.45 barA
# pt2 = 2.025 barA
# m_flow = 1.5 kg/s
#
# shape factor k = 0.9 [ k = 1 - (Rh1/Rs1)^2 ]
# swallowing capacity phi_t1 = 0.05
# outlet flow angle alpha2 = 65 deg
# pressure recovery factor diffuser Cp = 0.5
# ratio axial length to impeller outlet radius Lax/R2 = 0.7


import logging
from pint import Quantity
import matplotlib.pyplot as plt

from adet.assembly import CasadiSystem, solve_root_roblem
from adet.components import BladeRow
from adet.components.blade_row import VanelessDiffuser, plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork

from adet.equations.geometrical import (
    CompressorShapeFactor,
    LaxByOutradius,
    MinimalCamberLine,
)
from adet.equations.nondimensional import SwallowingCapacity, TotalTotalPressureRatio
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import (
    PercTotalPressureLoss,
    PercentageEntropyLoss,
    ZeroDeviation,
)

from adet.registries import GuessRegistry, ScalarsRegistry, VariableBoundsRegistry
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger

logger = logging.Logger(__name__)
setup_logger(logger)

# This makes the missing guesses default to 1
_greg = GuessRegistry()
_limreg = VariableBoundsRegistry()

_greg.set_fallback_value(1.0)
_greg.set('beta', -0.5)


NUM_SPAN = 1

# +++ Shafts
shaft = Shaft(
    omega=-1,
    is_constrained=False,
)

casing = Shaft(
    omega=Quantity(0, 'rpm'),
    is_constrained=True,
)

# +++ Fluid settings
fluid_model = ExternalFluidModel(
    DebugAbstractState('HEOS', 'Air'),  # This just counts the number of updates
)

fluid_settings = FluidSettings(
    model=fluid_model,
    update_variables=('p', 'T'),  # Thermodynamic iteration variables
    update_length=2,  # Single phase => Two update vars
)


# +++ Boundary conditions
inlet = Inlet(
    {
        'tot': {
            'p': Quantity(0.45, 'bar'),
            'T': Quantity(7.8, 'degC'),
        },
        'kin': {
            'relmach': 0.3,  # TODO: Use tip relmach 1.4
            'alpha': 0.0,
        },
        'oth': {
            'cum_massflow': 1.5,
        },
    },
)

ScalarsRegistry().set('shapeKCoeff', -1)

# +++ Components
rotor = BladeRow(
    name='rotor',
    shaft=shaft,
    in_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(0, 'deg'),
            'thick_by_pitch': 0.02,
            'shapeKCoeff': 0.9,
        },
    },
    out_constraints={
        'kin': {
            'alpha': Quantity(65, 'deg'),
        },
        'tot': {
            'p': Quantity(2.045, 'bar'),
        },
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(90, 'deg'),
            'thick_by_pitch': 0.02,
            'num_blades': 30,
        },
        'oth': {
            'chAx_outRad_Ratio': 0.7,
            'swllCap': 0.05,
        },
    },
    extra_equations={
        LaxByOutradius(): 1,
        MinimalCamberLine(): (0, 1),
        SwallowingCapacity(): (0, 1),
        CompressorShapeFactor(): 0,
        ZeroDeviation(): 0,  # Zero incidence
        ZeroDeviation(): 1,  # Zero Deviation
        # MinimalCamberLine(): (0, 1),
        PercentageEntropyLoss(0.0): (0, 1),
        # *** Definitions
        TotalTotalPressureRatio(): (0, 1),
    },
)

vaneless_diff = VanelessDiffuser(
    'diffuser',
    in_constraints={},
    out_constraints={
        'geo': {
            'rr_midspan': Quantity(0.3055659, 'm'),
            'heightRatio': 1.0,
        },
    },
    extra_equations={
        PercTotalPressureLoss(0.0): (0, 1),  # Isentropic
    },
)


ntw = ComponentNetwork(
    fluid_settings=fluid_settings,
    inlet=inlet,
    backend=CasadiSystem(num_span=NUM_SPAN),
    components=[rotor],
)

ntw.system.add_spanwise_constants('kin_Vm0')

ntw.build()

x0 = ntw.system.get_initial_guess()
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds()

# IPOPT is very robust, KINSOL faster but fails more easily
rootfinder = ntw.system.make_rootfinder(
    'kinsol',
    opts={'error_on_fail': True},
)
solution = solve_root_roblem(
    rootfinder,
    x0,
    kn,
    bnd,
    suppress_output=False,
)

ntw.system.write_solution_to_nodes(solution)

fig, axs = plt.subplots(2, 2, figsize=(8, 15))
for cmp_idx, cmp in enumerate(ntw.components):
    if cmp.inlet_node is None or cmp.outlet_node is None:
        raise ValueError('Missing nodes')

    node_idx = 0
    for n in (cmp.inlet_node, cmp.outlet_node):
        ax = axs[cmp_idx][node_idx]

        ax.set_title(f'Node number {2 * cmp_idx + node_idx}')
        ax.set_aspect('equal')
        n.kin.plot(n.geo, 8, ax)

        node_idx += 1


fig, ax = plt.subplots()
ax.set_aspect('equal')
offset = 0.0
for c in ntw.components:
    if not c.inlet_node or not c.outlet_node:
        raise ValueError('missing nodes')

    lines = plot_from_nodes(
        c.inlet_node,
        c.outlet_node,
        False,
        offset,
    )

    offset += c.outlet_node.geo.chord_ax[0]

for idx, n in enumerate(ntw.system.nodes):
    globals()[f'n{idx}'] = n


# plt.close('all')
plt.show()
