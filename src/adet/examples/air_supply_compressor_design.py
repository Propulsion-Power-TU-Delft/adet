# Design problem:
# Tt1 = 7.8 degC
# pt1 = 0.45 barA
# pt2 = 2.025 barA
# m_flow = 1.5 kg/s

# shape factor k = 1 - (Rh1/Rs1)^2 = 0.9
# swallowing capacity phi_t1 = 0.05
# outlet flow angle alpha2 = 65 deg
# pressure recovery factor diffuser Cp = 0.5
# ratio axial length to impeller outlet radius Lax/R2 = 0.7
#
# Stiffest models
# - Skin friction
# - Amirante - Explicit non-iterative (2015)


import logging
import scipy.stats.qmc as qmc
from pint import Quantity
import matplotlib.pyplot as plt

from adet.assembly import CasadiSystem, solve_root_roblem
from adet.components import BladeRow
from adet.components.blade_row import VanelessDiffuser, plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork

from adet.equations.definitions import IsentropicTotalEnthalpy
from adet.equations.geometrical import (
    LaxByOutradius,
    MinimalCamberLine,
)
from adet.equations.nondimensional import (
    GammaPV,
    WorkCoefficient,
    SwallowingCapacity,
    TotalTotalCompressionEfficiency,
)
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import (
    PercTotalPressureLoss,
    PercentageEntropyLoss,
    ZeroDeviation,
)

from adet.losses.compressors import (
    BackstromSlip,
    CaseyRushInletFunc,
    CompressorShapeFactor,
    EffectiveBladeNumber,
    LossAdder,
    ClearanceJansen,
    SkinFrictionJansen,
    BladeLoadingCoppage,
)
from adet.registries import GuessRegistry, ScalarsRegistry, VariableBoundsRegistry
from adet.tools.loggers import setup_logger
from adet.tools.coolprop_utils import DebugAbstractState

logger = logging.Logger(__name__)
setup_logger(logger)

# Set some bounds
_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
_bounds_reg.from_dict(
    {
        'massflow': (0.1, 4.0),
        'delta_hmass_.*': (10.0, 1e5),
        'delta_hmass_loading': (10.0, 1e4),  # This tends to diverge, bound it
    }
)

_greg = GuessRegistry()
_greg.reset()
_greg.from_dict(
    {
        'beta': -0.5,
        'gamma_pv': 1.4,
        'delta_hmass_.*': 1000,
    }
)
_greg.set_fallback_value(0.5)  # Missing values defaults to 0.5

NUM_SPAN = 1
ENABLE_LOSSES = False

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
    DebugAbstractState('REFPROP', 'Air'),  # This just counts the number of updates
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
            'alpha': 0.0,
        },
        'oth': {
            'massflow': 1.5,
        },
    },
)

ScalarsRegistry().set('chAx_outRad_Ratio', -1)

EQS_ISENTROPIC = {
    PercentageEntropyLoss(0.0): (0, 1),
    ZeroDeviation(): 1,
}
EQS_WITH_LOSSES = {
    BackstromSlip(): (0, 1),
    ClearanceJansen(): (0, 1),
    SkinFrictionJansen(): (0, 1),
    BladeLoadingCoppage(): (0, 1),
    LossAdder(): 1,
}

losses = EQS_ISENTROPIC if ENABLE_LOSSES else EQS_ISENTROPIC


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
            'tip_clearance': 1e-4,
        },
        'oth': {
            'swllCap': 0.05,
        },
    },
    out_constraints={
        'tot': {
            'p': Quantity(2.025, 'bar'),
        },
        'kin': {
            'alpha': Quantity(65, 'deg'),
        },
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(90, 'deg'),
            'thick_by_pitch': 0.02,
            'num_blades': 15,
            'num_splitters': 15,
        },
        'oth': {
            'workCoeff': 0.7,
            'chAx_outRad_Ratio': 0.7,
            # For losses
            'bl_loadingCoeff': 0.75,
            'abs_roughness': Quantity(0.01, 'mm'),
            'slip_factCoeff': 5.0,
        },
    },
    extra_equations={
        # *** Design parameters
        LaxByOutradius(): 1,
        SwallowingCapacity(): (0, 1),
        WorkCoefficient(): (0, 1),
        CompressorShapeFactor(): 0,
        CaseyRushInletFunc(): (0, 1),
        GammaPV(): 1,
        # *** Geometry
        MinimalCamberLine(): (0, 1),
        EffectiveBladeNumber(): 1,
        # *** Metal angle <-[link]-> Flow angle
        ZeroDeviation(): 0,  # Zero incidence
        # *** Enthalpy definitions
        IsentropicTotalEnthalpy(): (0, 1),
        TotalTotalCompressionEfficiency(): (0, 1),
        **losses,
    },
)

vaneless_diff = VanelessDiffuser(
    'diffuser',
    in_constraints={},
    out_constraints={
        'geo': {'heightRatio': 1.0},  # Constant diffuser height
        'oth': {'prFactor': 0.5},
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


ntw.build()

x0_is = ntw.system.get_initial_guess()
kn = ntw.system.get_scaled_constraints()
bnd_loss = ntw.system.get_arguments_bounds()

# IPOPT is very robust, KINSOL faster but fails more easily
rootfinder_is = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': True,
        'ipopt.tol': 1e-7,
        'ipopt.print_level': 3,
    },
)

# Get the isentropic solution, we will perturbate
# it to check the robustness of the solver
solution_is = solve_root_roblem(
    rootfinder_is,
    x0_is,
    kn,
    bnd_loss,
    suppress_output=False,
)

solution_is = solve_root_roblem(
    rootfinder_is,
    solution_is.tolist(),
    kn,
    bnd_loss,
    suppress_output=False,
)


sol_is_dict = ntw.system.solution_to_dict(solution_is)

# Remove isentropic and add losses
for eq, pos in EQS_ISENTROPIC.items():
    ntw.system.remove_equation(eq.__class__, pos)
for eq, pos in EQS_WITH_LOSSES.items():
    ntw.system.add_equation(eq, pos)

ntw.build()  # Rebuild

# Get
x0_loss = ntw.system.get_initial_guess(sol_is_dict)
bnd_loss = ntw.system.get_arguments_bounds()

rootfinder_loss = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.tol': 1e-10,
        'ipopt.print_level': 3,
    },
)

solution_loss = solve_root_roblem(
    rootfinder_loss,
    x0_loss,
    kn,
    bnd_loss,
    suppress_output=False,
)

ntw.system.write_solution_to_nodes(solution_loss)

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

# Assign nodes to globals for easier access
for idx, n in enumerate(ntw.system.nodes):
    globals()[f'n{idx}'] = n


# plt.close('all')
plt.show()
