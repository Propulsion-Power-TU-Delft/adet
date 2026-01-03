import logging
from pint import Quantity
import matplotlib.pyplot as plt
from adet.assembly import CasadiSystem, solve_root_roblem
from adet.components import BladeRow
from adet.components.blade_row import VanelessDiffuser, plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.geometrical import MinimalCamberLine
from adet.equations.nondimensional import (
    MidspanTotalTotalPressRatio,
    StaticStaticPressRatio,
    StaticTotalPressRatio,
    TotalTotalPressureRatio,
    WorkCoefficient,
)
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import (
    PercTotalPressureLoss,
    PercentageEntropyLoss,
    PlaceHolderLoss,
    ZeroDeviation,
    ZeroMidspanDeviation,
)

from adet.losses.compressors import (
    EndWallVelocities,
    IsentropicTotalEnthalpy,
    TotalTotalCompressionEfficiency,
)
from adet.registries import GuessRegistry, VariableBoundsRegistry
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger

_greg = GuessRegistry()
_breg = VariableBoundsRegistry()
_greg.set_fallback_value(1.0)

NUM_SPAN = 5

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
    DebugAbstractState('HEOS', 'Air'),
)

fluid_settings = FluidSettings(
    model=fluid_model,
    update_variables=('p', 'T'),  # Thermodynamic solver variables
    update_length=2,  # Single phase => Two update vars
)

logger = logging.Logger(__name__)
setup_logger(logger)

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
    uniform=True,
)


# +++ Components
rotor = BladeRow(
    name='rotor',
    shaft=shaft,
    in_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': Quantity(0.07416165, 'm'),
            'height': Quantity(0.0670433, 'm'),
            # *** Blades specs
            'metal_angle': Quantity(-44, 'deg'),
            'thick_by_pitch': 0.05,
        },
    },
    out_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(90, 'deg'),
            'rmid': Quantity(0.2159, 'm'),
            'height': Quantity(0.01524, 'm'),
            # *** Blades specs
            'metal_angle': Quantity(-30, 'deg'),
            'thick_by_pitch': 0.025,
            'chord_ax': Quantity(0.133879895, 'm'),
            'num_blades': 15,
        },
        'oth': {
            # 'eta_tt': 0.85,
            # 'pRatio_tt_midspan': 4.85,
        },
    },
    extra_equations={
        ZeroMidspanDeviation(): 1,  # = No deviation at midspan
        # ZeroDeviation(): 1,
        MinimalCamberLine(): (0, 1),
        PercentageEntropyLoss(0.0): (0, 1),  # Fully isentropic
        PlaceHolderLoss(): (0, 1),  # PLACEHOLDER
        # *** Enthalpy based losses
        IsentropicTotalEnthalpy(): (0, 1),
        TotalTotalCompressionEfficiency(): (0, 1),
        # *** Definitions
        EndWallVelocities(): 0,
        MidspanTotalTotalPressRatio(): (0, 1),
        StaticTotalPressRatio(): (0, 1),
        StaticStaticPressRatio(): (0, 1),
        TotalTotalPressureRatio(): (0, 1),
        WorkCoefficient(): (0, 1),
    },
)

vaneless_diff = VanelessDiffuser(
    'diffuser',
    in_constraints={},
    out_constraints={
        'geo': {
            'rmid': Quantity(0.3055659, 'm'),
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

ntw.system.add_spanwise_constants('stc_p1')

ntw.build()

x0 = ntw.system.get_initial_guess()
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds()

# IPOPT is very robust, KINSOL faster but fails more easily
rootfinder = ntw.system.make_rootfinder('ipopt')
solution = solve_root_roblem(
    rootfinder,
    x0,
    kn,
    bnd,
    suppress_output=False,
)

ntw.system.write_solution_to_nodes(solution)

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
        ax=ax,
    )

    offset += c.outlet_node.geo.chord_ax[0]

for idx, n in enumerate(ntw.system.nodes):
    n.kin.plot(n.geo, 14)
    plt.title(f'Node number {idx}')

print(ntw.components[0].outlet_node)

plt.show()
# plt.close('all')
