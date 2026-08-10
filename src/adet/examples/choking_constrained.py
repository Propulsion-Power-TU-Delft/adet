import logging

import matplotlib.pyplot as plt
import numpy as np
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
from adet.tools.plotting import setup_mpl
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
        # ThroatConditions(): idx - 1,
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
    **add_node(3),
}

BCS = {
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
    n1.geo.EffArea: 0.06,
    n2.geo.EffArea: 0.1,
    n3.geo.EffArea: 0.1,
    # n3.stc.Pressure: 1.88e5,
    n3.kin.FlowAngleRel: 0.0,
    # Throat
    # n0.geo.ThroatArea: 0.03,
    # n0.geo.ThroatRadius: 0.1,
    # n1.geo.ThroatArea: 0.03,
    # n1.geo.ThroatRadius: 0.1,
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
    (
        n0.kin.Omega,
        n1.kin.Omega,
        n2.kin.Omega,
        n3.kin.Omega,
    ),
    (
        n0.geo.RDistr,
        n1.geo.RDistr,
        n2.geo.RDistr,
        n3.geo.RDistr,
    ),
    (
        n0.kin.FlowAngleAbs,
        n1.kin.FlowAngleAbs,
        n2.kin.FlowAngleAbs,
    ),
)

system.add_boundary_conditions(BCS)

system.build()
input('Press enter to continue...')

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
        # Inlet Mach limit
        # n0.kin.MachThroat.Glob: (0.0, 1.01),
        n0.kin.RelMach: (0.0, 0.5),
        # n0.kin.MachThroat: (0.0, 1.0),
        n1.kin.RelMach: (0.9, 1.2),
        n2.kin.RelMach: (1.0, 5.0),
        n3.kin.RelMach: (1.0, 10.0),
    },
    ignore_defaults=True,
)

solution, optimizer = solve_optimization_problem(system, obj_func, x0, kn, bnd)

data = system.sol_to_dict(solution['x'].toarray().flatten())

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

if True:
    N_PTS = 30
    massflows = []
    out_machs = []
    p_ratios = []
    deviations = []
    mervels = []
    tanvels = []

    for deviation in np.linspace(0.0, 0.8, N_PTS):
        scale = system._scaling_manager.get_constraints_scaling()[-1]
        # kn[-1] = p_out / scale
        kn[-1] = np.array([deviation / scale])
        x0 = solution['x'].toarray()
        solution = optimizer(x0, kn)
        data = system.sol_to_dict(solution['x'].toarray().flatten())
        massflows.append(data[n0.oth.StreamMassFlow])
        p_ratios.append(data[n3.stc.Pressure] / data[n0.tot.Pressure])
        out_machs.append(data[n3.kin.RelMach])
        mervels.append(data[n3.kin.V_mer])
        tanvels.append(data[n3.kin.V_tan])
        deviations.append(
            np.abs(
                np.degrees(deviation),
            ),
        )

    setup_mpl(
        {
            'font.family': 'EB Garamond',
            'font.size': 20,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(p_ratios, deviations, linewidth=2, color='#8800bb')
    ax.grid(alpha=0.4)
    ax.set_xlabel(r'$p_{t,in} / p_{out}$')
    ax.set_ylabel(r'$\delta / \mathrm{[deg]}$')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(p_ratios, mervels, linewidth=2, color='#880022')
    ax.grid(alpha=0.4)
    ax.set_xlabel(r'$p_{t,in} / p_{out}$')
    ax.set_ylabel(r'$\rho_3V_{m,3}$')

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0.0, np.max(mervels))
    ax.set_ylim(0.0, np.max(tanvels))
    ax.set_aspect('equal')
    ax.set_xlabel(r'$V_m$')
    ax.set_ylabel(r'$V_t$')
    ax.grid(alpha=0.3)
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, N_PTS))
    for i, (vm, vt) in enumerate(zip(mervels, tanvels)):
        ax.quiver(0, 0, vm, vt, scale_units='xy', scale=1, angles='xy', color=colors[i])

    plt.show()
