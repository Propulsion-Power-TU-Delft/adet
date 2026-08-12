"""
Choking massflow prediction as a massflow maximisation problem. This
should mirror the lagrangian example formulation.

Note:
-----
The massflow across the non choked sections can collapse to
either the subsonic or supersonic brach. This does not influence
the choking massflow.
"""

from adet.equations.control_volumes import SimpleThroat

import logging

from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.equations.fundamental import (
    EulerEquation,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.equations.nondimensional import RelativeMachNumber, StaticTotalPressRatio
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink
from adet.solution import solve_optimization_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

system = CasadiSystem(1)
n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)


def add_node(idx: int):
    return {  # *** Node 1
        Kinematics(): idx,
        MassAreaRelation(): idx,
        TotalStaticMatching(): idx,
        SimpleThroat(): idx - 1,
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
    StaticTotalPressRatio(): (0, 3),
    **add_node(1),
    **add_node(2),
}

BCS = {
    n0.tot.Pressure: 20e5,
    n0.tot.Temperature: 700,
    # *** Constants
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.kin.Omega: 0,
    n0.geo.RDistr: 0.1,
    n0.geo.ThroatRadius: 0.1,
    n1.geo.ThroatRadius: 0.1,
    # ***
    # Area distribution
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

system.fluid_settings = FluidSettings(
    fluid_state=abs_state,
    update_variables=(n0.stc.Pressure.Glob, n0.stc.Temperature.Glob),
)


[system.add_equation(eq, pos) for eq, pos in EQS.items()]
system.add_equalities(
    (n0.kin.Omega, n1.kin.Omega, n2.kin.Omega),
    (n0.geo.RDistr, n1.geo.RDistr, n2.geo.RDistr),
    (n0.kin.FlowAngleAbs, n1.kin.FlowAngleAbs, n2.kin.FlowAngleAbs),
)

system.add_boundary_conditions(BCS)

system.build()

# *** Optimizer formulation
obj_func = 1 / system.free_args_sym[n1.oth.StreamMassFlow]
# ***

x0 = system.get_guess()
kn = system.get_boundary_conds()
bnd = system.get_bounds(
    {
        # Node limiters
        n0.stc.Pressure.Glob: (1, 1e7),
        n0.stc.Temperature.Glob: (150, 1e4),
        # Throat limiters
        n0.oth.ThrPressure.Glob: (1, 1e7),
        n0.oth.ThrTemperature.Glob: (150, 1e4),
        # Mach limit
        n0.kin.MachThroat.Glob: (0.0, 1.01),
        n0.kin.RelMach.Glob: (0.0, 0.9),
    },
    ignore_defaults=True,
)

solution, optimizer = solve_optimization_problem(system, obj_func, x0, kn, bnd)

data = system.sol_to_dict(solution['x'].toarray().flatten())

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
