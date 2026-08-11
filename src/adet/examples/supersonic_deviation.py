"""
Work in progress example of an equation-oriented formulation of
subsonic to supersonic treatment of deviation downstream of an axial blade row
"""

# === IMPORTS
import logging
from copy import deepcopy
from typing import Literal

import ipdb  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.blade_row import RowGeometry
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.geometrical import MeridionalGeometry  # noqa: F401
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.losses.mixing import (
    AungierDeviationModel,
    SieverdingBasePressure,
)
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


INLET_PTOT = 10e5
INLET_TEMPERATURE = 600
# *** Inlet conditions
inlet = Inlet(
    boundary_conditions={
        # *** Inlet geometry
        n0.kin.FlowAngleAbs: 0.0,
        n0.geo.Rmid: 0.1,
        n0.geo.Area: 0.1,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        # *** Inlet total conditions
        n0.tot.Pressure: INLET_PTOT,
        n0.tot.Temperature: INLET_TEMPERATURE,
    }
)


casing = Shaft(0, is_constrained=True)
shaft = Shaft(3000, is_constrained=True)


stat_blade = BladeRow(
    name='stator',
    shaft=casing,
    bound_cond={
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.geo.NumBlades: 10,
        # 0 == Blade/B.L. thickness
        n0.geo.ThickByPitch: 0.1,
        # n0.oth.DispByMom: 2,
        # n0.oth.MomByBld: 0.08,
        # n0.oth.DispByHgt: 0.0,
        # 1 ==
        n1.geo.ThickByPitch: 0.0,
        # Blade chord
        n1.geo.ChordAx: 0.05,
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # Flow aligned with throat
        IsentropicLink(): (0, 1),
        # BoundaryLayerRatios(): 0,
    },
    constant_variables=[n0.geo.Rmid, n0.geo.Height],  # Constant mean radius
)

stat_mix = deepcopy(stat_blade)
stat_mix.name = 'st_mix'


stat_blade.set_bc_from_dict(
    {
        n1.kin.FlowAngleRel: Quantity(60, 'deg'),
    }
)
stat_mix.set_bc_from_dict(
    {
        n1.oth.StreamMassFlow: 20,
    }
)

# Remove isentropic
stat_mix.remove_equation(IsentropicLink, (0, 1))

# Add mixing equations
# stat_mix.add_equation(MixingMomentumBalances(), (0, 1))
stat_mix.add_equation(AungierDeviationModel(), (0, 1))
# Base pressure correlation
stat_mix.add_equation(SieverdingBasePressure(), (0, 1))

# Rotor
rot_blade = deepcopy(stat_blade)
rot_blade.set_bc_from_dict(
    {
        n0.kin.FlowAngleRel: Quantity(40, 'deg'),
        n1.kin.FlowAngleRel: Quantity(-40, 'deg'),
    }
)

rot_blade.shaft = shaft
#
rot_mix = deepcopy(stat_mix)
rot_mix.shaft = shaft


stat_blade.set_spanwise_constant(
    # Uniform inlet
    n0.kin.V_mer,
    n0.geo.HDistr,
    # Uniform chords along the span
    n1.geo.ChordAx,
)

# *** Network buildup
fluid_settings = FluidSettings(
    fluid_state=abs_state,
    update_variables=(n0.stc.Pressure, n0.stc.Temperature),
)

ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(1),
    [
        stat_blade,
        stat_mix,
        # rot_blade,
        # rot_mix,
    ],
)

MODE: Literal['root', 'opt'] = 'root'

if MODE == 'root':
    ntw.system.add_boundary_conditions({n3.stc.Pressure: 8e5})

ntw.build()
# input('Continue?')

# *** Solution
x0 = ntw.system.get_guess(fallback=0.8)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds(
    {
        n0.kin.RelMach: (0.0, 1.0),
        # n1.kin.RelMach: (0.0, 1.0),
        # n0.geo.Chord.Glob: (0.0, 1e5),
        n0.stc.Pressure.Glob: (10.0, 1.1 * INLET_PTOT),
        n0.stc.Temperature.Glob: (60.0, 1.1 * INLET_TEMPERATURE),
    },
    ignore_defaults=False,
)


if MODE == 'opt':
    obj_func = 1 / ntw.system.free_args_sym[n1.oth.MassFlow]
    sol, opt = solve_optimization_problem(ntw.system, obj_func, x0, kn, bnd)
    sol = sol['x'].toarray()


elif MODE == 'root':
    # Ipopt rootfinding
    try:
        # Unbounded
        rtfn = ntw.system.make_rootfinder(
            'ipopt',
            {
                'error_on_fail': True,
                'ipopt.max_wall_time': 5,
            },
        )
        sol = solve_root_problem(rtfn, x0, kn, suppress_output=False)
    except RuntimeError:
        # Bounded
        rtfn = ntw.system.make_rootfinder('ipopt', {'error_on_fail': False})
        sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)

    # kinsol
    rtfn = ntw.system.make_rootfinder('kinsol')
    sol = solve_root_problem(rtfn, sol, kn, suppress_output=True)

data = ntw.system.sol_to_dict(sol)


setup_mpl({'font.size': 20})
SWEEP = False
if SWEEP:
    N_PTS = 60
    devs = []
    p_outs_plot = []
    mach_thrs = []
    mach_n1 = []
    entropies = []
    mass_flows = []
    rtfn = ntw.system.make_rootfinder('kinsol', {'error_on_fail': True})
    out_node = n1 if len(ntw.components) == 1 else n3

    p_outs = np.linspace(0.8 * INLET_PTOT, 0.528 * INLET_PTOT, N_PTS)

    for idx, po in enumerate(p_outs):
        kn[-1] = np.array([po]) / ntw.system.constraints_scaling[-1]
        try:
            sol = solve_root_problem(rtfn, sol, kn)
            pass
        except RuntimeError:
            rtfn = ntw.system.make_rootfinder(
                'ipopt',
                {
                    'error_on_fail': False,
                    'ipopt.max_wall_time': 5,
                },
            )
            sol = solve_root_problem(rtfn, sol, kn, bnd)
            data = ntw.system.sol_to_dict(sol)
            try:
                sol = solve_root_problem(rtfn, sol, kn)
                rtfn = ntw.system.make_rootfinder('kinsol', {'error_on_fail': True})
                sol = solve_root_problem(rtfn, sol, kn)
            except RuntimeError:
                break

        if data[n3.kin.FlowAngleRel] >= 1.5:
            break
        data = ntw.system.sol_to_dict(sol)
        p_outs_plot.append(po)
        devs.append(np.abs(data[n1.geo.MetalAngle] - data[out_node.kin.FlowAngleRel]))
        # devs.append(data[n3.kin.DevAngle])
        mach_thrs.append(data[out_node.kin.RelMach])
        mach_n1.append(data[n1.kin.RelMach])
        entropies.append(data[out_node.loss.Ds_mixing])
        mass_flows.append(data[n1.oth.MassFlow])

    # Calculate pressure ratios
    pr = np.array(p_outs_plot) / INLET_PTOT

    fig, ((ax, ax_pr), (ax_m1, ax_mf)) = plt.subplots(2, 2, figsize=(16, 12))

    # First subplot: deviation and Mach number
    ax.plot(p_outs_plot, devs, 'C0', label=r'$\delta$')
    ax.set_xlabel(r'$p_{out}$ / [Pa]')
    ax.set_ylabel(r'$\delta$ / [deg]', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')

    ax2 = ax.twinx()
    ax2.plot(p_outs_plot, mach_thrs, 'C1', label=r'$M_{out}$')
    ax2.set_ylabel(r'$M_{out}$ / [-]', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    # Add crosshair at sonic point (M = 1)
    mach_idx = np.argmin(np.abs(np.array(mach_thrs) - 1.0))
    p_sonic = p_outs_plot[mach_idx]
    ax.axvline(
        x=p_sonic,
        color='gray',
        linestyle='--',
        alpha=0.5,
        linewidth=1,
    )
    ax2.axhline(
        y=1.0,
        color='gray',
        linestyle='--',
        alpha=0.5,
        linewidth=1,
    )

    # Second subplot: outlet flow angle vs pressure ratio
    ax_pr.plot(pr, entropies, 'C2')
    ax_pr.set_xlabel(r'$p_{out}/p_{t,in}$ / [-]')
    ax_pr.set_ylabel(r'$\Delta s_{\mathrm{mix}}$ / $\mathrm{[Jkg^{-1}K^{-1}]}$')
    ax_pr.grid(alpha=0.3)

    # Third subplot: Mach number at n1 vs outlet pressure
    ax_m1.plot(p_outs_plot, mach_n1, 'C3')
    ax_m1.set_xlabel(r'$p_{out}$ / [Pa]')
    ax_m1.set_ylabel(r'$M_{\mathrm{throat}}$ / [-]')
    ax_m1.grid(alpha=0.3)

    # Fourth subplot: mass flow vs outlet pressure
    ax_mf.plot(p_outs_plot, mass_flows, 'C4')
    ax_mf.set_xlabel(r'$p_{out}$ / [Pa]')
    ax_mf.set_ylabel(r'$\dot{m}$ / [kg/s]')
    ax_mf.grid(alpha=0.3)

    fig.tight_layout()
    fig.show()

# globals().update(residual_debugger(MixingMomentumBalances(), [2, 3], data))

PLOTS = True
if PLOTS:
    if len(ntw.components) == 1:
        n3 = n1
        n2 = n0
    # *** Velocity triangles
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )  # sharex=True, sharey=True)
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
    plt.show()
