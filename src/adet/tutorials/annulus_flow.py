from adet.varspec import NodeStates
from adet.equations.utils import residual_debugger
from adet.fluid.ideal_eos import IdealGasState
import logging
import casadi as cs

import numpy as np
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalMassFlow,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas, MeridionalGeometry
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.fluid.settings import FluidSettings
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

EQUATIONS = {
    AnnulusAreas(): 0,  # A_geo = 2 pi r H
    ZeroBlockage(): 0,  # No blockage (A_eff = A_geo)
    MassAreaRelation(): 0,  # m_dot = rho V A_eff
    AbsoluteMachNumber(): 0,  # Defines Mach number
    TotalStaticMatching(): 0,  # Matches total and static state
    Kinematics(): 0,  # Defines velocity triangles
    TotalMassFlow(): 0,  # Total massflow across all streamtubes
    MeridionalGeometry(): 0,  # Geometry of the annulus
}

# Fundamental entities
system = CasadiSystem(num_span=1)
node0 = NodeVariables(0)

# *** Fluid model
ideal_state = IdealGasState(1.4, 287.8, 2e-5)
# ideal_state = DebugAbstractState('HEOS', 'air')

# ***
fluid_settings = FluidSettings(
    fluid_state=ideal_state,
    update_variables=(node0.stc.Pressure, node0.stc.Temperature),
)
system.fluid_settings = fluid_settings

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

BC = {
    node0.kin.Omega: 1000.0,
    node0.kin.FlowAngleAbs: Quantity(30, 'deg'),  # Inlet absolute flow angle
    node0.oth.TotMassFlow: 100.0,  # Total massflow across the annulus
    node0.geo.Rmid: 0.1,  # Midpoint annulus radius
    node0.geo.Height: 0.1,  # Annulus height
    node0.geo.MeridionalAngle: 0.0,
    node0.tot.Pressure: 18.1e5,
    node0.tot.Temperature: 573.15,
}

system.add_boundary_conditions(BC)
system.add_spanwise_constants(node0.kin.V_mer, node0.geo.HDistr)

system.build()

rtfn = system.make_rootfinder(
    'kinsol',
    {
        'error_on_fail': False,
        # 'print_iteration': True,
        'print_level': 2,
    },
)

x0 = system.get_guess(fallback=0.5)
kn = system.get_boundary_conds()
resfunc = system.make_residual_function()

# newton iterations
x = x0
for _ in range(4):
    func_val = resfunc(x, kn)

    if any(np.isnan(func_val)):
        break

    jac_val = resfunc.jacobian()(x, kn, func_val)
    step = cs.inv(jac_val[0]) @ func_val
    x -= 0.7 * step
    hes_val = resfunc.jacobian().jacobian()(x, kn, func_val, *jac_val)

    print(f'Function values are {func_val}\n')
    print(f'Temperature {x[11]} \n')
    print(f'Step is {step}\n')


sol = solve_root_problem(rtfn, x, kn)

sol_dict = system.sol_to_dict(x.toarray())

globals().update(
    residual_debugger(ideal_state.get_eos(NodeStates.STATIC), [0], sol_dict)
)
