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
    AnnulusAreas(): 0,  # A_geo = 2 pi r H
    ZeroBlockage(): 0,  # No blockage (A_eff = A_geo)
    MassAreaRelation(): 0,  # m_dot = rho V A_eff
    AbsoluteMachNumber(): 0,  # Define Mach number
    TotalStaticMatching(): 0,  # Matches total and static state
    Kinematics(): 0,  # Defines velocity triangles
}

# Fundamental entities
thrm = ThermoVariables()
system = CasadiSystem()
node0 = NodeVariables(0)

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
    node0.kin.Omega: 1000.0,
    node0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    node0.oth.MassFlow: 100.0,
    node0.geo.RDistr: 0.1,
    node0.geo.HDistr: 0.1,
    node0.tot.Pressure: 18.1e5,
    node0.tot.Temperature: 573.15,
}

system.add_boundary_conditions(BC)
system.build()

rtfn = system.make_rootfinder('kinsol')

x0 = system.get_guess()
kn = system.get_boundary_conds()

sol = solve_root_problem(rtfn, x0, kn)

sol_dict = system.sol_to_dict(sol)
