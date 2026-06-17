from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import logging

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.equations.base_equation import EquationBase
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
    **add_node(3),
    **add_node(4),
}

BCS = {
    # n0.oth.TgtMassFlow: 109,
    n0.tot.Pressure: 1e6,
    n0.tot.Temperature: 500,
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.kin.Omega: 0,
    n0.geo.RDistr: 0.1,
    # Areas
    n0.geo.EffArea: 0.1,
    n1.geo.EffArea: 0.06,
    # Pressure
    n4.geo.EffArea: 0.1,
    n4.kin.FlowAngleAbs: 0.6,
    n4.stc.Pressure: 6e5,
}

# NOTE: Nozzle scheme
#               Shock
#  ______        .......
#        \     /`||
#         \___/  ||
#  _ . _ . _ . _ || . _
#
#     |     |   |  |   |
#     0     1   2  3   4

abs_state = DebugAbstractState('HEOS', 'Air')
fluid_model = FluidModel(abs_state)

system.fluid_settings = FluidSettings(
    fluid_model,
    (n0.stc.Pressure.Glob, n0.stc.Temperature.Glob),
)


class NormalShock(EquationBase):
    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
        ds: n1.loss.Ds_shock.Hint,
        p0: n0.stc.Pressure.Hint,
        rho0: n0.stc.Density.Hint,
        p1: n1.stc.Pressure.Hint,
        rho1: n1.stc.Density.Hint,
        w0: n0.kin.V_mag.Hint,
        w1: n1.kin.V_mag.Hint,
    ):
        r1 = ds - (s1 - s0)
        r2 = (p0 + rho0 * w0**2) - (p1 + rho1 * w1**2)
        r3 = (1 - (p0 - p1) / (2 * p0)) * 0.4
        return r1, r2


[system.add_equation(eq, pos) for eq, pos in EQS.items()]
system.remove_equation(IsentropicLink, (2, 3))
system.add_equation(NormalShock(), (2, 3))

system.add_equalities(
    (
        n0.kin.Omega,
        n1.kin.Omega,
        n2.kin.Omega,
        n3.kin.Omega,
        n4.kin.Omega,
    ),
    (
        n0.geo.RDistr,
        n1.geo.RDistr,
        n2.geo.RDistr,
        n3.geo.RDistr,
        n4.geo.RDistr,
    ),
    (
        n0.kin.FlowAngleAbs,
        n1.kin.FlowAngleAbs,
        n2.kin.FlowAngleAbs,
        n3.kin.FlowAngleAbs,
        # n4.kin.FlowAngleAbs,
    ),
    (
        n2.geo.EffArea,
        n3.geo.EffArea,
    ),
)

system.add_boundary_conditions(BCS)

system.build()
input('Press enter to continue...')

# *** Optimizer formulation
obj_func = 1 / system.free_args_sym[n1.oth.MassFlow]
# ***

x0 = system.get_scaled_guess(fallback=0.8)
kn = system.get_scaled_constraints()
bnd = system.get_arguments_bounds(
    {
        # Node limiters
        n0.stc.Pressure.Glob: (1, 1e7),
        n0.stc.Temperature.Glob: (110, 1e4),
        # Throat limiters
        # Inlet Mach limit
        # n0.kin.MachThroat.Glob: (0.0, 1.01),
        n0.kin.RelMach: (0.0, 0.9),
        n2.kin.RelMach: (1.0, 10.0),
        n3.geo.EffArea: (0.06, 0.1),
        n3.loss.Ds_shock: (0.0, 1e10),
    },
    ignore_defaults=True,
)

solution, optimizer = solve_optimization_problem(
    system,
    obj_func,
    x0,
    kn,
    bnd,
    {'error_on_fail': True},
)

sol = solution['x'].toarray().flatten()
# rtfn = system.make_rootfinder('kinsol')
# solve_root_problem(rtfn, sol, kn)

data = system.sol_to_dict(sol)

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
setup_mpl(
    {
        'font.family': 'EB Garamond',
        'font.size': 20,
    }
)
SWEEP = True
if SWEEP:
    N_PTS = 15
    massflows = []
    out_machs = []
    p_ratios = []
    deviations = []
    mervels = []
    tanvels = []

    fig_m, ax_m = plt.subplots(figsize=(8, 8))
    SPACE = np.linspace(6e5, 9.0e5, N_PTS)

    cmap = plt.get_cmap('plasma')
    norm = Normalize(SPACE[0], SPACE[-1])
    for press in SPACE:
        color = cmap(norm(press))
        scale = system.constraints_scaling[-1]
        kn[-1] = np.array([press / scale])
        x0 = solution['x'].toarray()
        try:
            solution = optimizer(x0, kn)
            ax_m.plot(
                [
                    0,
                    1 / 2,
                    data[n3.geo.EffArea][0] / data[n4.geo.EffArea][0],
                    data[n3.geo.EffArea][0] / data[n4.geo.EffArea][0],
                    1,
                ],
                [
                    data[n0.kin.RelMach][0],
                    data[n1.kin.RelMach][0],
                    data[n2.kin.RelMach][0],
                    data[n3.kin.RelMach][0],
                    data[n4.kin.RelMach][0],
                ],
                'o-',
                color=color,
            )
        except RuntimeError:
            continue
        data = system.sol_to_dict(solution['x'].toarray().flatten())
        massflows.append(data[n0.oth.MassFlow])
        out_machs.append(data[n1.kin.RelMach])
        # mervels.append(data[n3.kin.V_mer])
        # tanvels.append(data[n3.kin.V_tan])

    # Plots machs
    ax_m.set_xlabel(r'$x/l$ / [-]')
    ax_m.set_ylabel(r'$M$ / [-]')
    ax_m.grid(alpha=0.5)

    cb = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax_m)
    cb.set_label(r'$p_{\mathrm{out}}$')

    fig_m.show()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    ax1.plot(SPACE, massflows, linewidth=2, color='#8800bb')
    ax1.grid(alpha=0.4)
    ax1.set_ylabel(r'$\dot{m} / \mathrm{[kgs^{-1}]}$')

    ax2.plot(SPACE, out_machs, linewidth=2, color='#880022')
    ax2.grid(alpha=0.4)
    ax2.set_xlabel(r'$p_3 / \mathrm{[Pa]}$')
    ax2.set_ylabel(r'$M_3$')

    # fig, ax = plt.subplots(figsize=(8, 3))
    # ax.set_xlim(0.0, np.max(mervels))
    # ax.set_ylim(0.0, np.max(tanvels))
    # ax.set_aspect('equal')
    # ax.set_xlabel(r'$V_m$')
    # ax.set_ylabel(r'$V_t$')
    # ax.grid(alpha=0.3)
    # cmap = plt.get_cmap('plasma')
    # colors = cmap(np.linspace(0, 1, N_PTS))
    # for i, (vm, vt) in enumerate(zip(mervels, tanvels)):
    #     ax.quiver(0, 0, vm, vt, scale_units='xy', scale=1, angles='xy', color=colors[i])

    plt.show()
