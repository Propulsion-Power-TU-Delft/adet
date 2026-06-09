import logging

from ambiance import Atmosphere
from CoolProp import AbstractState
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components.blade_row import BladeRow
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.fluid.settings import FluidModel, FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

std_atm = Atmosphere(10e3)
abs_state = AbstractState('HEOS', 'air')

shaft = Shaft(
    Quantity(5000, 'rpm'),
    is_constrained=True,
)

inlet = Inlet(
    {
        'stc': {
            'p': std_atm.pressure,
            'T': std_atm.temperature,
        },
        'kin': {
            'mach': 0.6,
            'alpha': 0.0,
        },
        'oth': {
            'massflow': 80,
        },
    }
)

fan_blade = BladeRow(
    'fan',
    shaft,
    'rotor',
    in_constraints={
        'geo': {
            'meridional_angle': 0.0,
            'thick_by_pitch': 0.0,
        },
    },
    out_constraints={
        'stc': {
            'p': 1.2 * std_atm.pressure,
        },
        'geo': {
            'hubtipRatio': 0.3,
            'heightRatio': 1.0,  # Assume constant channel height
            'meridional_angle': 0.0,
            # Irrelevant
            'bld_thick': 1,
            'num_blades': 1,
            'chord_ax': 1,
        },
    },
    extra_equations={
        ZeroDeviation(): 0,
        IsentropicLink(): (0, 1),
    },
    constant_variables=['geo_rr_midspan'],
)

fluid_model = FluidModel(abs_state)

settings = FluidSettings(fluid_model, ('p', 'T'), 2)

ntw = ComponentNetwork(
    settings,
    inlet,
    CasadiSystem(1),
    [fan_blade],
)

ntw.build()

rtfn = ntw.system.make_rootfinder('ipopt')

x0 = ntw.system.get_scaled_guess()
kn = ntw.system.get_scaled_constraints()

solution = solve_root_problem(rtfn, x0, kn)

ntw.system.write_solution_to_nodes(solution)

n0 = ntw.system.nodes[0]
n1 = ntw.system.nodes[1]
