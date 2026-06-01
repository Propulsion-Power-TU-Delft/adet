import logging

import casadi as cs
import numpy as np

from adet.assembly import IPOPT_DEFAULTS, CasadiSystem
from adet.equations.fundamental import (
    EulerEquation,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import IsentropicLink
from adet.registries import ScalingRegistry
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

ScalingRegistry()['K * kg * s / m**2'] = 1000

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

res_expr = res_func(
    free_args_symbols,
    constraints_symbols,
)

# Manual Lagrangian choking formulation
mf = system.free_args_sym[n0.oth.MassFlow]
lamb = cs.MX.sym('lambda', max(res_expr.shape))

# Objective function is massflow
lagrangian = mf + cs.dot(lamb, res_expr)
grad_lagrangian = cs.gradient(lagrangian, free_args_symbols)

full_residual = cs.vertcat(res_expr, grad_lagrangian)
full_variables = cs.vertcat(free_args_symbols, lamb)


opt_problem = {
    'x': full_variables,  # Free args (with multipliers)
    'p': constraints_symbols,  # Knowns
    'g': full_residual,  # Equality constraints = 0
}

optimizer = cs.nlpsol('optimizer', 'ipopt', opt_problem, IPOPT_DEFAULTS)

x0 = np.concatenate(
    (system.get_scaled_guess(), np.zeros(lamb.shape)),
)
kn = np.concatenate(system.get_scaled_constraints())
bnd = system.get_arguments_bounds()

kwargs = {
    # Force the root problem
    'lbg': 0,
    'ubg': 0,
    # Free variables limits + far limit on multipliers
    'lbx': np.concatenate(
        (bnd[0], -1e20 * np.ones(lamb.shape)),
    ),
    'ubx': np.concatenate(
        (bnd[1], +1e20 * np.ones(lamb.shape)),
    ),
}

solution = optimizer(x0=x0, p=kn, **kwargs)

sol_data = system.sol_to_dict(solution['x'].toarray().flatten())
