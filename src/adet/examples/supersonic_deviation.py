# === IMPORTS
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.blade_row import RowGeometry
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.fluid.settings import FluidModel, FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.solution import solve_optimization_problem, solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_camberline, plot_velocity_triangles, setup_mpl
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)
logger = logging.getLogger(__name__)
setup_logger(logger)

abs_state = DebugAbstractState('HEOS', 'Air')

# *** Inlet conditions
inlet = Inlet(
    boundary_conditions={
        # *** Inlet geometry
        # n0.oth.MassFlow: 227.5,
        n0.kin.FlowAngleAbs: 0.0,
        n0.geo.Rmid: 0.1,
        n0.geo.Area: 0.1,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        # *** Inlet total conditions
        n0.tot.Pressure: 20e5,
        n0.tot.Temperature: 500,
    }
)


casing = Shaft(0, is_constrained=True)
shaft = Shaft(0, is_constrained=False)


stat_conv = BladeRow(
    name='stator',
    shaft=casing,
    bound_cond={
        n0.geo.ThickByPitch: 0.0,
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.geo.ThickByPitch: 0.0,
        n1.geo.AspectRatio: 3.0,
        # n1.geo.FlareAngle: Quantity(30, 'deg'),
        n1.geo.ClearanceByHeight: 0.01,
        n1.geo.NumBlades: 10,  # Dummy input
        # n1.geo.ZweifelCoeff: 0.9,
        # *** Boundary layer
        n1.oth.MomByBld: 0.075,
        n1.oth.DispByMom: 2,
        n1.oth.DispByHgt: 0.05,
        # *** Losses
        n1.oth.CdProfile: 0.002,
        n1.oth.XiCambLenA: 0.375,
        n1.oth.XiCambLenB: 0.675,
        n1.oth.DischCoeff: 0.35,
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # No deviation
        IsentropicLink(): (0, 1),
        # ModifiedZweifel(): (0, 1),
    },
    constant_variables=[n0.geo.Rmid],
)

stat_div = deepcopy(stat_conv)
stat_div.name = 'st_throat'
stat_div.set_component_constants(n0.geo.Height)


stat_conv.set_bc_from_dict(
    {
        n1.geo.Area: 0.11,
        n1.kin.FlowAngleRel: Quantity(55, 'deg'),
        # n1.kin.RelMach: 1.0,
    }
)

stat_div.set_bc_from_dict(
    {
        # n1.kin.RelMach: 1.5,
        # n1.kin.FlowAngleRel: 0.0,
        # n1.stc.Pressure: 15e5,
    }
)

# stat_div.remove_equation(IsentropicLink, (0, 1))
# stat_div.add_equation(AungierDeviationModel(), (0, 1))


fluid_model = FluidModel(abs_state)
fluid_settings = FluidSettings(
    fluid_model,
    (n0.stc.Pressure, n0.stc.Temperature),
)

ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(1),
    [
        stat_conv,
        stat_div,
    ],
)

stat_conv.set_spanwise_constant(
    # Uniform inlet
    n0.kin.V_mer,
    n0.geo.HDistr,
    # Uniform chords along the span
    n1.geo.ChordAx,
)

ntw.system.add_equalities((n2.stc.Pressure, n3.stc.Pressure))
ntw.build()
input('Continue?')


x0 = ntw.system.get_scaled_guess(fallback=0.8)
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds(
    {
        n0.kin.RelMach: (0.0, 1.0),
        n1.kin.RelMach: (0.0, 1.0),
        # n0.geo.Chord.Glob: (0.0, 1e5),
        n0.stc.Pressure.Glob: (10.0, 25e5),
        n0.stc.Temperature.Glob: (60.0, 500),
    },
    ignore_defaults=False,
)

obj_func = 1 / ntw.system.free_args_sym[n1.oth.MassFlow]
sol, opt = solve_optimization_problem(ntw.system, obj_func, x0, kn, bnd)
sol = sol['x'].toarray()


# # Ipopt rootfinding
# try:
#     # Unbounded
#     rtfn = ntw.system.make_rootfinder(
#         'ipopt', {'error_on_fail': True, 'max_wall_time': 10}
#     )
#     sol = solve_root_problem(rtfn, x0, kn, suppress_output=False)
# except RuntimeError:
#     # Bounded
#     rtfn = ntw.system.make_rootfinder('ipopt', {'error_on_fail': False})
#     sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)
#
#
# # Kinsol
# rtfn = ntw.system.make_rootfinder('kinsol')
# sol = solve_root_problem(rtfn, sol, kn)

data = ntw.system.sol_to_dict(sol)


LOOP = False
if LOOP:
    N_PTS = 100
    devs = []
    machs = np.linspace(1.0, 2.5, N_PTS)
    for mach in machs:
        kn[-1] = np.array([mach])
        sol = solve_root_problem(rtfn, sol, kn)
        data = ntw.system.sol_to_dict(sol)
        devs.append(data[n3.kin.DevAngle])

    fig, ax = plt.subplots()
    ax.plot(machs, devs)
# ax.plot(machs, data[n1.geo.MetalAngle] * np.ones(N_PTS))

PLOTS = False
if PLOTS:
    setup_mpl({'font.family': 'EB Garamond', 'font.size': 15})

    # *** Velocity triangles
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for ax, node in zip(axs.flatten(), [n0, n1, n2, n3]):
        ax.set_aspect('equal')
        plot_velocity_triangles(
            data[node.kin.V_tan],
            data[node.kin.V_mer],
            data[node.kin.BladeSpeed],
            data[node.geo.RDistr],
            ax,
        )

    fig_mer, ax_mer = plt.subplots()
    fig_cbl, ax_cbl = plt.subplots()

    # Setup axes
    ax_mer.set_ylim(0.0, 1.01 * data[n3.geo.Rtip])
    ax_mer.set_aspect('equal')
    ax_cbl.set_aspect('equal')
    ax_mer.grid(alpha=0.4)
    ax_cbl.grid(alpha=0.4)

    offset = 0
    for nodes in [(n0, n1), (n2, n3)]:
        geom = RowGeometry(
            data[nodes[0].geo.Rmid][0],
            data[nodes[0].geo.Rmid][0],
            data[nodes[0].geo.Height][0],
            data[nodes[1].geo.Height][0],
            data[nodes[0].geo.MeridionalAngle][0],
            data[nodes[1].geo.MeridionalAngle][0],
            data[nodes[1].geo.ChordAx][0],
            axial_offset=offset,
        )
        plot_camberline(
            data[nodes[0].geo.MetalAngle],
            data[nodes[1].geo.MetalAngle],
            data[nodes[1].geo.ChordAx],
            ax=ax_cbl,
            color='k',
            axial_offset=offset,
        )
        # opt_pitch = (
        #     2 * np.pi * data[nodes[1].geo.Rmid] / data[nodes[1].geo.NumBladesOpt]
        # )
        plot_camberline(
            data[nodes[0].geo.MetalAngle],
            data[nodes[1].geo.MetalAngle],
            data[nodes[1].geo.ChordAx],
            ax=ax_cbl,
            color='k',
            axial_offset=offset,
            tangential_offset=data[n1.geo.Pitch],
        )
        geom.plot_meridional_profile(ax=ax_mer, color='k')
        # Add offset
        offset += data[nodes[1].geo.ChordAx][0]

    # *** Camber lines

    plt.tight_layout()
    plt.show()
