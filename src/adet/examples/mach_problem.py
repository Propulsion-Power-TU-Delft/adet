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
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

EQUATIONS = {
    AnnulusAreas(): 0,  # A = 2 pi r H
    MassAreaRelation(): 0,  # m_dot = rho V A
    AbsoluteMachNumber(): 0,  # Define Mach number
    TotalStaticMatching(): 0,  # Matches total and static state
    ZeroBlockage(): 0,  # No blockage in passage
    Kinematics(): 0,  # Defines angles velocity
}

# Fundamental entities
thrm = ThermoVariables()
system = CasadiSystem()
node_0 = NodeVariables(0)

# *** Fluid model
ideal_state = IdealGasState(1.4, 287, 2e-5)
# ***
fluid_settings = FluidSettings(
    fluid_state=ideal_state,
    update_variables=(thrm.Pressure, thrm.Temperature),
)
system.fluid_settings = fluid_settings

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

BC = {
    node_0.kin.Omega: 0.0,
    node_0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    node_0.oth.MassFlow: 0.132,
    node_0.geo.RDistr: 0.038,
    node_0.geo.HDistr: 0.002,
    node_0.tot.Pressure: 18.1e5,
    node_0.tot.Temperature: 573.15,
}

system.add_boundary_conditions(BC)
system.build()

rtfn = system.make_rootfinder('kinsol')

x0 = system.get_guess()
kn = system.get_boundary_conds()

sol = solve_root_problem(rtfn, x0, kn)

sol_dict = system.sol_to_dict(sol)
