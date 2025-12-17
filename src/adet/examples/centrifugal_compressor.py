from pint import Quantity
import matplotlib.pyplot as plt
from adet.assembly import CasadiSystem, solve_problem
from adet.components import BladeRow
from adet.components.blade_row import plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.geometrical import NoCamberline
from adet.equations.nondimensional import StaticTotalPressRatio, TotalTotalPressureRatio
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.registries import DefaultUnitsRegistry
from adet.tools.coolprop_utils import DebugAbstractState


# Components definition

shaft = Shaft(
    omega=Quantity(-1000, 'rpm'),
    is_constrained=True,
)

inlet = Inlet(
    {
        'tot': {
            'p': 1e5,
            'T': 300,
        },
        'oth': {
            'cum_massflow': 100,
        },
        'kin': {
            'alpha': 0.0,
        },
    }
)

rotor = BladeRow(
    'rotor',
    shaft=shaft,
    in_constraints={
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': Quantity(1, 'm'),
            'height': Quantity(0.5, 'm'),
        },
    },
    out_constraints={
        'geo': {
            'meridional_angle': Quantity(90, 'deg'),
            'heightRatio': 0.25,
            'metal_angle': Quantity(80, 'deg'),
            'chord_ax': Quantity(2.5, 'm'),
        },
        'oth': {
            'STratio': 3,
        },
    },
    extra_equations={
        ZeroDeviation(): 0,  # = No incidence
        ZeroDeviation(): 1,  # = No deviation
        NoCamberline(): (0, 1),  # = Don't compute any camber geometry
        StaticTotalPressRatio(): (0, 1),
        PercentageEntropyLoss(0.0): (0, 1),  # Isentropic
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
    backend=CasadiSystem(spanwise_stations=1),
    components=[rotor],
)

# Set custom units and defaults
_dfu_reg = DefaultUnitsRegistry()

# Add units for custom variables
_dfu_reg.from_dict(
    {
        'STratio': 'dimensionless',
    }
)

ntw.system.build()
rootfinder = ntw.system.make_rootfinder('nlpsol')

x0 = ntw.system.get_initial_guess()
kn = ntw.system.get_scaled_constraints()

solution = solve_problem(rootfinder, x0, kn)

ntw.system.write_solution_to_nodes(solution)

fig, ax = plt.subplots()
ax.set_aspect('equal')
lines = plot_from_nodes(
    ntw.system.nodes[0],
    ntw.system.nodes[1],
    False,
    0.0,
    ax=ax,
)
fig.show()
