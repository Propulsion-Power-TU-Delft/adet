from pint import Quantity
import matplotlib.pyplot as plt
from adet.assembly import CasadiSystem, solve_problem
from adet.components import BladeRow
from adet.components.blade_row import VanelessDiffuser, plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.geometrical import MinimalCamberLine
from adet.equations.nondimensional import (
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
)

from adet.losses.compressors import EndWallVelocities, TotalTotalCompressionEfficiency
from adet.registries import GuessRegistry
from adet.tools.coolprop_utils import DebugAbstractState

_greg = GuessRegistry()
_greg.set_fallback_value(1.0)

shaft = Shaft(
    omega=Quantity(21789, 'rpm'),
    is_constrained=True,
)

casing = Shaft(
    omega=Quantity(0, 'rpm'),
    is_constrained=True,
)

inlet = Inlet(
    {
        'tot': {
            'p': 101352.9,
            'T': 288.16,
        },
        'oth': {
            'cum_massflow': 4.989512,
        },
    },
    uniform=True,
)

rotor = BladeRow(
    'rotor',
    shaft=shaft,
    in_constraints={
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': Quantity(0.07416165, 'm'),
            'height': Quantity(0.0670433, 'm'),
            'metal_angle': Quantity(-44, 'deg'),
            'thick_by_pitch': 0.05,
        },
    },
    out_constraints={
        'geo': {
            'meridional_angle': Quantity(90, 'deg'),
            'rmid': Quantity(0.2159, 'm'),
            'height': Quantity(0.01524, 'm'),
            'metal_angle': Quantity(-30, 'deg'),
            # *** Blades specs
            'thick_by_pitch': 0.025,
            'chord_ax': Quantity(0.133879895, 'm'),
            'num_blades': 15,
        },
        # 'oth': {'eta_tt': 0.87},
    },
    extra_equations={
        ZeroDeviation(): 0,  # = No incidence
        ZeroDeviation(): 1,  # = No deviation
        MinimalCamberLine(): (0, 1),
        PercentageEntropyLoss(0.0): (0, 1),
        TotalTotalCompressionEfficiency(): (0, 1),
        # ***
        EndWallVelocities(): 0,
        PlaceHolderLoss(): (0, 1),  # Isentropic
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


fluid_model = ExternalFluidModel(
    DebugAbstractState('HEOS', 'Air'),
)

fluid_settings = FluidSettings(
    model=fluid_model,
    update_variables=('p', 'T'),
    update_length=2,
)

ntw = ComponentNetwork(
    fluid_settings=fluid_settings,
    inlet=inlet,
    backend=CasadiSystem(num_span=3),
    components=[rotor],
)

ntw.build()

rootfinder = ntw.system.make_rootfinder('ipopt')

x0 = ntw.system.get_initial_guess()
kn = ntw.system.get_scaled_constraints()

solution = solve_problem(rootfinder, x0, kn)

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
