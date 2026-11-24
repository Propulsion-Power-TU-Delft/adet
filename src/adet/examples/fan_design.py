"""
Fan design example using the node interface.

Demonstrates a simple axial fan blade row with isentropic flow
and zero deviation. Inlet conditions are set at 10km altitude with
Mach 0.6, and the blade row produces a pressure rise of 20%.
"""

import logging

from ambiance import Atmosphere
from CoolProp import AbstractState
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.components.blade_row import BladeRow
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.nondimensional import TotalTotalPressureRatio
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

n0 = NodeVariables(0)
n1 = NodeVariables(1)

std_atm = Atmosphere(10e3)
abs_state = AbstractState('HEOS', 'air')

shaft = Shaft(
    Quantity(5000, 'rpm'),
    is_constrained=True,
)

inlet = Inlet(
    boundary_conditions={
        n0.stc.Pressure: std_atm.pressure,
        n0.stc.Temperature: std_atm.temperature,
        n0.kin.Mach: 0.6,
        n0.kin.FlowAngleAbs: Quantity(0.0, 'deg'),
        n0.oth.TotMassFlow: 80,
    }
)

fan_blade = BladeRow(
    name='fan',
    shaft=shaft,
    bound_cond={
        # Purely axial
        n0.geo.MeridionalAngle: 0.0,
        n1.geo.MeridionalAngle: 0.0,
        # Chord and number of blades
        n1.geo.AspectRatio: 6.0,
        n1.geo.Solidity: 1.0,
        # Meridional channel
        n1.geo.HubTipRatio: 0.3,
        n1.geo.HeightRatio: 0.95,
        # Pressure ratio
        n1.ndim.PRatioTT: 1.2,
    },
    extra_equations={
        ZeroDeviation(): 0,
        IsentropicLink(): (0, 1),
        TotalTotalPressureRatio(): (0, 1),
    },
    constant_variables=[n0.geo.Rmid],
)

settings = FluidSettings(
    fluid_state=abs_state,
    update_variables=(n0.stc.Pressure, n0.stc.Temperature),
)

ntw = ComponentNetwork(
    settings,
    inlet,
    CasadiSystem(1),
    [fan_blade],
)

ntw.build()

rtfn = ntw.system.make_rootfinder('ipopt')
x0 = ntw.system.get_guess(fallback=0.5)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds()

solution = solve_root_problem(rtfn, x0, kn)
sol_dict = ntw.system.sol_to_dict(solution)
