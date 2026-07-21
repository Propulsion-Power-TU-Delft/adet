import logging

from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.nondimensional import AbsoluteMachNumber, GammaIdeal
from adet.equations.special import ThermoVarsAdder
from adet.fluid.settings import FluidSettings
from adet.fluid.ideal_eos import IdealGasState
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

EQUATIONS = {
    TotalStaticMatching(): 0,
    AnnulusAreas(): 0,
    MassAreaRelation(): 0,
    AbsoluteMachNumber(): 0,
    ZeroBlockage(): 0,  # Area = Eff area
    Kinematics(): 0,
    ThermoVarsAdder(): 0,
    # GammaPV(): 0,
    GammaIdeal(): 0,
}

system = CasadiSystem()
# *** Fluid model
abs_state = DebugAbstractState('REFPROP', 'MM')
ideal_state = IdealGasState(1.4, 287, 2e-5)
# ***
thrm = ThermoVariables()
fluid_settings = FluidSettings(
    fluid_state=ideal_state,
    update_variables=(thrm.Pressure, thrm.Temperature),
)
system.fluid_settings = fluid_settings

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

n0 = NodeVariables(0)
BC = {
    n0.kin.Omega: 0.0,
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.oth.MassFlow: 0.132,
    n0.geo.RDistr: 0.038,
    n0.geo.HDistr: 0.002,
    n0.tot.Pressure: 18.1e5,
    n0.tot.Temperature: 573.15,
}

system.add_boundary_conditions(BC)
system.build()

rtfn = system.make_rootfinder('kinsol')

x0 = system.get_guess()
kn = system.get_boundary_conds()

sol = solve_root_problem(rtfn, x0, kn)

sol_dict = system.sol_to_dict(sol)
