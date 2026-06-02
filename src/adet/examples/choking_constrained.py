import logging

import casadi as cs
import numpy as np

from adet.assembly import IPOPT_DEFAULTS, CasadiSystem
from adet.equations.base_equation import EquationBase
from adet.equations.control_volumes import ThroatConditions
from adet.equations.fundamental import (
    EulerEquation,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.equations.utils import safe_if_else, safe_max, safe_min
from adet.fluid.settings import ExternalFluidModel, FluidSettings
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
        ThroatConditions(): (idx - 1, idx),
        # LimitedMach(): idx,
        AbsoluteMachNumber(): idx,
        # --- 0 -> 1
        EulerEquation(): (idx - 1, idx),
        MassConservation(): (idx - 1, idx),
        IsentropicLink(): (idx - 1, idx),
    }


class LimitedMach(EquationBase):
    def residual(
        self,
        mach_th: n0.kin.MachThroat.Hint,
        a1: n0.stc.SpeedSound.Hint,
        W1: n0.kin.W_mag.Hint,
        mach1: n0.kin.Mach.Hint,
    ):

        res_lim = W1 - safe_max(1.0 * a1, mach1 * a1)
        res_unl = W1 - mach1 * a1

        return safe_if_else(mach_th >= 0.99, res_lim, res_unl)


class LimitedMassflow(EquationBase):
    def residual(
        self,
        mf_target: n0.oth.TgtMassFlow.Hint,
        mf_actual: n0.oth.MassFlow.Hint,
        mf_choke: n0.oth.ChokeMassflow.Hint,
    ):

        return mf_actual - safe_min(mf_target, mf_choke)


EQS = {
    # *** Node 0
    # LimitedMassflow(): 0,
    Kinematics(): 0,
    MassAreaRelation(): 0,
    AbsoluteMachNumber(): 0,
    TotalStaticMatching(): 0,
    **add_node(1),
    **add_node(2),
}

BCS = {
    # n0.oth.ChokeMassflow: 108.2,
    # n0.oth.TgtMassFlow: 109,
    n0.tot.Pressure: 20e5,
    n0.tot.Temperature: 500,
    n0.kin.FlowAngleAbs: 0,
    n0.oth.MassFlow: 10.0,
    n0.kin.Omega: 0,
    n0.geo.RDistr: 0.1,
    # Areas
    n0.geo.EffArea: 0.1,
    n0.geo.ThroatArea: 0.05,
    n1.geo.EffArea: 0.08,
    n1.geo.ThroatArea: 0.03,
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
fluid_model = ExternalFluidModel(abs_state)

system.fluid_settings = FluidSettings(
    fluid_model,
    (n0.stc.Pressure.Glob, n0.stc.Temperature.Glob),
)


[system.add_equation(eq, pos) for eq, pos in EQS.items()]
system.add_equalities(
    (n0.kin.Omega, n1.kin.Omega, n2.kin.Omega),
    (n0.geo.RDistr, n1.geo.RDistr, n2.geo.RDistr),
    (n0.kin.FlowAngleAbs, n1.kin.FlowAngleAbs, n2.kin.FlowAngleAbs),
    (n0.oth.ChokeMassflow, n1.oth.ChokeMassflow),
    (n0.oth.RltEnthalpyChoke, n0.rlt.Enthalpy),  # Inlet choke
    (n0.oth.RltEnthalpyChoke, n1.oth.RltEnthalpyChoke),
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

mf = system.free_args_sym[n1.oth.ChokeMassflow]

opt_problem = {
    'x': free_args_symbols,
    'p': constraints_symbols,
    'f': 1 / mf,
    'g': res_expr,
}

optimizer = cs.nlpsol(
    'optimizer',
    'ipopt',
    opt_problem,
    {**IPOPT_DEFAULTS, 'error_on_fail': False},
)

x0 = np.concatenate(system.get_scaled_guess())
kn = np.concatenate(system.get_scaled_constraints())
bnd = system.get_arguments_bounds(
    {
        n0.stc.Pressure.Glob: (1e2, 30e5),
        n0.stc.Temperature.Glob: (150.0, 1e4),
        n0.oth.ThrPressure.Glob: (1e2, 30e5),
        n0.oth.ThrTemperature.Glob: (150.0, 1e4),
    },
    ignore_defaults=True,
)

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
