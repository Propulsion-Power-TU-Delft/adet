from adet.equations.nondimensional import AbsoluteMachNumber
import logging

import casadi as cs
import numpy as np

from adet.assembly import CasadiSystem, IPOPT_DEFAULTS
from adet.equations.fundamental import (
    EulerEquation,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import IsentropicLink
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

system = CasadiSystem(1)
n0 = NodeVariables(0)
n1 = NodeVariables(1)

EQS = {
    EulerEquation(): (0, 1),
    MassConservation(): (0, 1),
    IsentropicLink(): (0, 1),
    MassAreaRelation(): 0,
    MassAreaRelation(): 1,
    Kinematics(): 0,
    Kinematics(): 1,
    TotalStaticMatching(): 0,
    TotalStaticMatching(): 1,
    AbsoluteMachNumber(): 0,
    AbsoluteMachNumber(): 1,
}

BCS = {
    n0.tot.Pressure: 20e5,
    n0.tot.Temperature: 300,
    n0.kin.FlowAngleAbs: 0,
    # n0.oth.MassFlow: 10.0,
    n0.kin.Omega: 0,
    n0.geo.RDistr: 0.1,
    n0.geo.EffArea: 0.1,
    n1.geo.EffArea: 0.02,
}

abs_state = DebugAbstractState('HEOS', 'Air')
fluid_model = ExternalFluidModel(abs_state)

system.fluid_settings = FluidSettings(
    fluid_model,
    (n0.stc.Pressure.Glob, n0.stc.Temperature.Glob),
)


[system.add_equation(eq, pos) for eq, pos in EQS.items()]
system.add_equalities(
    (n0.kin.Omega, n1.kin.Omega),
    (n0.geo.RDistr, n1.geo.RDistr),
    (n0.kin.FlowAngleAbs, n1.kin.FlowAngleAbs),
)

system.add_boundary_conditions(BCS)

system.build()

res_func = system.make_residual_function()

args_sym = list(system.free_args_sym.values())
cons_sym = list(system.const_sym.values())

free_args_symbols = cs.vertcat(*args_sym)
constraints_symbols = cs.vertcat(*cons_sym)

res_expr_partial = res_func(
    free_args_symbols,
    constraints_symbols,
)

mf = system.free_args_sym[n0.oth.MassFlow]

opt_problem = {
    'x': free_args_symbols,
    'p': constraints_symbols,
    'f': 1 / mf,
    'g': res_expr_partial,
}

optimizer = cs.nlpsol('optimizer', 'ipopt', opt_problem, IPOPT_DEFAULTS)

x0 = np.concatenate(system.get_scaled_guess())
kn = np.concatenate(system.get_scaled_constraints())
bnd = system.get_arguments_bounds()

kwargs = {
    # Force the root problem
    'lbg': 0,
    'ubg': 0,
    # Free variables limits
    'lbx': bnd[0],
    'ubx': bnd[1],
}

solution = optimizer(x0=x0, p=kn, **kwargs)

sol_data = system.sol_to_dict(solution['x'].toarray().flatten())
