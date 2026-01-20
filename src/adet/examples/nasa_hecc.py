import logging
from pint import Quantity
import matplotlib.pyplot as plt

from adet.assembly import CasadiSystem, solve_root_roblem
from adet.components import BladeRow
from adet.components.blade_row import VanelessDiffuser, plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork

from adet.equations.definitions import IsentropicTotalEnthalpy
from adet.equations.fundamental import BladeBlockage
from adet.equations.geometrical import (
    EndwallProperties,
    MinimalCamberLine,
    MeridionalVariable,
)
from adet.equations.nondimensional import (
    GammaPV,
    SwallowingCapacity,
    WorkCoefficient,
    StaticTotalPressRatio,
    StaticStaticPressRatio,
    TotalTotalPressureRatio,
    TotalTotalCompressionEfficiency,
)
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import (
    PercTotalPressureLoss,
    PercentageEntropyLoss,
    PlaceHolderLoss,
    ZeroDeviation,
)

from adet.losses.compressors import (
    BackstromSlip,
    BladeLoadingCoppage,
    ClearanceJansen,
    LossAdder,
    SkinFrictionJansen,
)
from adet.registries import GuessRegistry, VariableBoundsRegistry
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger

logger = logging.Logger(__name__)
setup_logger(logger)

# This makes the missing guesses default to 1
_greg = GuessRegistry()
_limreg = VariableBoundsRegistry()
_limreg.set('Vm', (1.0, 300.0))

_greg.set_fallback_value(0.8)
_greg.set('beta', -0.5)

NUM_SPAN = 7

# +++ Shafts
shaft = Shaft(
    omega=Quantity(21000, 'rpm'),
    is_constrained=True,
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
            'p': 101352.9,
            'T': 288.16,
        },
        'kin': {
            'alpha': 0.0,
        },
        'oth': {
            'cum_massflow': 4.989512,
        },
    },
)


# +++ Components
rotor = BladeRow(
    name='rotor',
    shaft=shaft,
    in_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': Quantity(0.07416165, 'm'),
            'height': Quantity(0.0670433, 'm'),
            # *** Blades specs
            'metal_angle': Quantity(-44, 'deg'),
            'thick_by_pitch': 0.02,
            'tip_clearance': 1e-3,
        },
    },
    out_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(90, 'deg'),
            'rr_midspan': Quantity(0.2159, 'm'),
            'height': Quantity(0.01524, 'm'),
            # *** Blades specs
            'metal_angle': Quantity(-30, 'deg'),
            'thick_by_pitch': 0.02,  # Thickness by pitch ratio
            'chord_ax': Quantity(0.133879895, 'm'),
            'num_blades': 30,
        },
        'oth': {
            # 'eta_tt': 0.87,  # Total total efficiency
            # For losses
            'slip_factCoeff': 5.0,
            'abs_roughness': Quantity(0.05, 'mm'),
            'bl_loadingCoeff': 0.75,
        },
    },
    extra_equations={
        MeridionalVariable(): 1,
        BackstromSlip(): (0, 1),
        # ZeroDeviation(): 1,
        MinimalCamberLine(): (0, 1),
        # PercentageEntropyLoss(0.0): (0, 1),
        PlaceHolderLoss(): (0, 1),
        # *** Enthalpy based Losses
        IsentropicTotalEnthalpy(): (0, 1),
        TotalTotalCompressionEfficiency(): (0, 1),
        BladeLoadingCoppage(): (0, 1),
        ClearanceJansen(): (0, 1),
        WorkCoefficient(): (0, 1),
        SkinFrictionJansen(): (0, 1),
        TotalTotalPressureRatio(): (0, 1),
        LossAdder(): 1,
        # *** Blockage (optional)
        # BladeBlockage(): 0,
        # BladeBlockage(): 1,
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
    backend=CasadiSystem(num_span=NUM_SPAN, scale_suffix='<|'),
    components=[rotor],
)

ntw.system.add_spanwise_constants('kin_Vm0')
ntw.system.add_spanwise_constants('stc_p1')

ntw.build()

x0 = ntw.system.get_initial_guess()
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds()

# IPOPT is more robust, takes variable limits into account -> For 'bi-stable' solutions
# KINSOL is faster, sometimes converges on problems where ipopt struggles
rootfinder = ntw.system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False},
)
solution = solve_root_roblem(
    rootfinder,
    x0,
    kn,
    bnd,
    suppress_output=False,
)

ntw.system.write_solution_to_nodes(solution)

fig, axs = plt.subplots(2, 2, figsize=(8, 20))
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
