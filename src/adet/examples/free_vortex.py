from adet.solution import solve_root_problem
import logging

from pint import Quantity

from adet.assembly import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    TotalStaticMatching,
    FreeVortexDistribution,
    MassAreaRelation,
    ZeroBlockage,
)
from adet.equations.geometrical import (
    MeridionalGeometry,
    EndwallProperties,
    AnnulusAreas,
)
from adet.fluid.settings import FluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

system = CasadiSystem(num_span=3)
n0 = NodeVariables(0)

EQS = {
    # *** Node 0
    Kinematics(): 0,
    TotalStaticMatching(): 0,
    MeridionalGeometry(): 0,
    EndwallProperties(): 0,
    FreeVortexDistribution(): 0,
    MassAreaRelation(): 0,
    AnnulusAreas(): 0,
    ZeroBlockage(): 0,
}

BCS = {
    n0.tot.Pressure: 1e6,
    n0.tot.Temperature: 500,
    # Kine
    n0.kin.Omega: 0,
    n0.kin.V_mer: 30,
    n0.kin.Beta_mid: Quantity(20, 'deg'),
    # Geometry
    n0.geo.Rmid: 0.1,
    n0.geo.Height: 0.1,
    n0.geo.MeridionalAngle: 0.0,
}
abs_state = DebugAbstractState('HEOS', 'Air')
idl_state = IdealGasState(1.4, 287, 2e-5)

fluid_model = FluidModel(abs_state)

system.fluid_settings = FluidSettings(
    fluid_model,
    (n0.stc.Pressure.Glob, n0.stc.Temperature.Glob),
)
system.add_spanwise_constants(n0.oth.MassFlow)


[system.add_equation(eq, pos) for eq, pos in EQS.items()]
system.add_boundary_conditions(BCS)

system.build()
input('Press enter to continue...')


x0 = system.get_scaled_guess(fallback=0.01)
kn = system.get_scaled_constraints()
bnd = system.get_arguments_bounds(
    {
        # Node limiters
        n0.stc.Pressure.Glob: (1, 1e7),
        n0.stc.Temperature.Glob: (110, 1e4),
    },
    ignore_defaults=False,
)

rtfn = system.make_rootfinder('kinsol', {'error_on_fail': False})
sol = solve_root_problem(rtfn, x0, kn, bnd)

data = system.sol_to_dict(sol)
