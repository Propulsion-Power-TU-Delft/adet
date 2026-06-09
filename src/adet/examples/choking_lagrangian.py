import logging

import casadi as cs
import numpy as np
from pint import Quantity

from adet.assembly import IPOPT_DEFAULTS, CasadiSystem
from adet.equations.control_volumes import ThroatConditions
from adet.equations.fundamental import (
    EulerEquation,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.equations.nondimensional import RelativeMachNumber
from adet.fluid.settings import FluidModel, FluidSettings
from adet.losses.basic import IsentropicLink
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

system = CasadiSystem(1)
# Some nodes
n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)
n4 = NodeVariables(4)
n5 = NodeVariables(5)
n6 = NodeVariables(6)
n7 = NodeVariables(7)
n8 = NodeVariables(8)


def add_node(idx: int):
    return {  # *** Node 1
        Kinematics(): idx,
        MassAreaRelation(): idx,
        TotalStaticMatching(): idx,
        ThroatConditions(): idx - 1,
        RelativeMachNumber(): idx,
        # --- 0 -> 1
        EulerEquation(): (idx - 1, idx),
        MassConservation(): (idx - 1, idx),
        IsentropicLink(): (idx - 1, idx),
    }


EQS = {
    # *** Node 0
    Kinematics(): 0,
    MassAreaRelation(): 0,
    RelativeMachNumber(): 0,
    TotalStaticMatching(): 0,
    **add_node(1),
    **add_node(2),
}

BCS = {
    # n0.oth.ChokeMassflow: 108.2,
    # n0.oth.TgtMassFlow: 109,
    n0.tot.Pressure: 20e5,
    n0.tot.Temperature: 500,
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    # n1.kin.FlowAngleRel: Quantity(-20, 'deg'),
    # n0.oth.MassFlow: 10.0,
    n0.kin.Omega: 0,
    n0.geo.RDistr: 0.1,
    # Areas
    n0.geo.EffArea: 0.1,
    n0.geo.ThroatArea: 0.05,
    n0.geo.ThroatRadius: 0.1,
    n1.geo.EffArea: 0.08,
    n1.geo.ThroatArea: 0.03,
    n1.geo.ThroatRadius: 0.1,
    n2.geo.EffArea: 0.07,
}

#  ______                 /`````
#        \     /`````\___/
#         \___/
#  _ . _ . _ . _ . _ . _ . _ . _ .
#
#     |     |     |    |     |
#
#     0     th    1    th    2

abs_state = DebugAbstractState('HEOS', 'Air')
fluid_model = FluidModel(abs_state)

system.fluid_settings = FluidSettings(
    fluid_model,
    (n0.stc.Pressure.Glob, n0.stc.Temperature.Glob),
)


[system.add_equation(eq, pos) for eq, pos in EQS.items()]
system.add_equalities(
    (n0.kin.Omega, n1.kin.Omega, n2.kin.Omega),
    (n0.geo.RDistr, n1.geo.RDistr, n2.geo.RDistr),
    (n0.kin.FlowAngleAbs, n1.kin.FlowAngleAbs, n2.kin.FlowAngleAbs),
)

system.add_boundary_conditions(BCS)

system.build()
input('Press enter to continue...')

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
mf = system.free_args_sym[n1.oth.MassFlow]
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

optimizer = cs.nlpsol(
    'optimizer',
    'ipopt',
    opt_problem,
    {**IPOPT_DEFAULTS, 'error_on_fail': False},
)

x0 = np.concatenate(
    (system.get_scaled_guess(), np.ones(lamb.shape)),
)
kn = np.concatenate(system.get_scaled_constraints())
bnd = system.get_arguments_bounds(
    {
        # Node limiters
        n0.stc.Pressure.Glob: (1, 1e7),
        n0.stc.Temperature.Glob: (150, 1e4),
        # Throat limiters
        n0.oth.ThrPressure.Glob: (1, 1e7),
        n0.oth.ThrTemperature.Glob: (150, 1e4),
        # Inlet Mach limit
        # n0.kin.MachThroat.Glob: (0.0, 1.01),
        n0.kin.RelMach: (0.0, 1.0),
    },
    ignore_defaults=True,
)

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

rtfn = cs.rootfinder('rtfn', 'kinsol', opt_problem)
rtfn(x0=solution['x'], p=kn)

sol_data = system.sol_to_dict(solution['x'].toarray().flatten())
