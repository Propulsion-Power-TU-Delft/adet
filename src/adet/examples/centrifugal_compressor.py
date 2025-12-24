from pint import Quantity
import matplotlib.pyplot as plt
from adet.assembly import CasadiSystem, solve_problem
from adet.components import BladeRow
from adet.components.blade_row import VanelessDiffuser, plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.geometrical import MinimalCamberLine
from adet.equations.nondimensional import (
    StaticTotalPressRatio,
    TotalTotalPressureRatio,
    WorkCoefficient,
)
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import PercentageEntropyLoss, TotalPressureLoss, ZeroDeviation
from adet.registries import DefaultUnitsRegistry
from adet.tools.coolprop_utils import DebugAbstractState

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
    }
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
            'chord_ax': Quantity(0.133879895, 'm'),
            'num_blades': 15,
            'thick_by_pitch': 0.025,
        },
    },
    extra_equations={
        ZeroDeviation(): 0,  # = No incidence
        ZeroDeviation(): 1,  # = No deviation
        MinimalCamberLine(): (0, 1),
        StaticTotalPressRatio(): (0, 1),
        WorkCoefficient(): (0, 1),
        PercentageEntropyLoss(0.0): (0, 1),  # Isentropic
    },
)

vaneless_diff = VanelessDiffuser(
    'diffuser',
    in_constraints={},
    out_constraints={
        'geo': {
            'rmid': Quantity(0.25, 'm'),
            'heightRatio': 1.0,
        },
    },
    extra_equations={
        # TotalPressureLoss(0.0): (0, 1),
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
    backend=CasadiSystem(num_span=1),
    components=[rotor, vaneless_diff],
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
rootfinder = ntw.system.make_rootfinder('ipopt')

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

for n in ntw.system.nodes:
    n.kin.plot(n.geo, 14)

plt.show()
