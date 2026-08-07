# === IMPORTS
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from CoolProp import AbstractState
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.blade_row import RowGeometry
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.definitions import RepeatedStage
from adet.equations.fundamental import FreeVortexDistribution
from adet.equations.geometrical import ModifiedZweifel
from adet.equations.nondimensional import (
    FlowCoefficientMid,
    StaticTotalDegreeOfReaction,
    TotalTotalExpansionEfficiency,
    WorkCoefficientMid,
)
from adet.fluid.settings import FluidSettings
from adet.losses.basic import (
    TotalPressureLoss,
    ZeroDeviation,
)
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_camberline, plot_velocity_triangles, setup_mpl
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)

logger = logging.getLogger(__name__)
setup_logger(logger)

abs_state = AbstractState('HEOS', 'Air')

# *** Inlet conditions
inlet = Inlet(
    boundary_conditions={
        # *** Inlet geometry
        n0.oth.CumMassFlow: 10.0,
        n0.geo.Rmid: 0.1,
        n0.geo.HubTipRatio: 0.65,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        # *** Inlet total conditions
        n0.tot.Pressure: 10e5,
        n0.tot.Temperature: 500,
    }
)


casing = Shaft(0, is_constrained=True)
shaft = Shaft(-1, is_constrained=False)

stator = BladeRow(
    name='stator',
    shaft=casing,
    bound_cond={
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.geo.AspectRatio: 3.0,
        n1.geo.NumBlades: 40,  # Number of blades
        n1.geo.ZweifelCoeff: 0.9,  # Theoretical n_bl,opt
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # No deviation
        TotalPressureLoss(0.9): (0, 1),  # Loss coefficient
        ModifiedZweifel(): (0, 1),
    },
    constant_variables=[n0.geo.Rmid],
    spanwise_constants=[n1.geo.ChordAx],
)

# > Modify the rotor
rotor = deepcopy(stator)  # Reuse the stator as template
rotor.shaft = shaft  # Assign the rotating shaft
rotor.name = 'rotor'

stator.set_spanwise_constant(
    # Uniform inlet meridional velocity and streamtubes heights
    n0.kin.V_mer,
    n0.geo.HDistr,
)

fluid_settings = FluidSettings(
    fluid_state=abs_state,
    update_variables=(n0.stc.Pressure, n0.stc.Temperature),
)

ntw = ComponentNetwork(
    fluid_settings=fluid_settings,
    inlet=inlet,
    backend=CasadiSystem(num_span=1),
    components=[stator, rotor],
)

ntw.system.add_equation(FlowCoefficientMid(), (0, 3))
ntw.system.add_equation(WorkCoefficientMid(), (0, 3))
ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))  # eta_tt


rotor.set_bc_from_dict(
    {
        n1.ndim.FlowCoeffMid: 0.4,
        n1.ndim.WorkCoeffMid: -1.1,
        n1.ndim.DegreeOfReactionTS: 0.6,
    }
)

# Free vortex radial equilibrium
if ntw.system.num_span > 1:
    stator.add_equation(FreeVortexDistribution(), 1)
    rotor.add_equation(FreeVortexDistribution(), 1)

ntw.build()

x0 = ntw.system.get_guess(fallback=0.8)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds(
    {
        n0.geo.Chord.Glob: (0.0, 1e5),
        n0.kin.V_mag.Glob: (0.0, 400.0),
        n0.stc.Pressure.Glob: (10.0, 13e5),
        n0.stc.Temperature.Glob: (60.0, 500),
    },
    ignore_defaults=False,
)

# Ipopt
try:
    # Unbounded
    rtfn = ntw.system.make_rootfinder(
        'ipopt', {'error_on_fail': True, 'ipopt.max_wall_time': 5}
    )
    sol = solve_root_problem(rtfn, x0, kn, suppress_output=False)
except RuntimeError:
    # Bounded
    rtfn = ntw.system.make_rootfinder('ipopt', {'error_on_fail': True})
    sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)


# Kinsol
# rtfn = ntw.system.make_rootfinder('kinsol')
# sol = solve_root_problem(rtfn, sol, kn)

sol_dict = ntw.system.sol_to_dict(sol)

PLOTS = True
if PLOTS:
    setup_mpl({'font.family': 'EB Garamond', 'font.size': 15})

    # *** Velocity triangles
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for ax, node in zip(axs.flatten(), [n0, n1, n2, n3]):
        ax.set_aspect('equal')
        plot_velocity_triangles(
            sol_dict[node.kin.V_tan],
            sol_dict[node.kin.V_mer],
            sol_dict[node.kin.BladeSpeed],
            sol_dict[node.geo.RDistr],
            ax,
        )

    fig_mer, ax_mer = plt.subplots()
    fig_cbl, ax_cbl = plt.subplots()

    # Setup axes
    ax_mer.set_ylim(0.0, 1.01 * sol_dict[n3.geo.Rtip])
    ax_mer.set_aspect('equal')
    ax_cbl.set_aspect('equal')
    ax_mer.grid(alpha=0.4)
    ax_cbl.grid(alpha=0.4)

    offset = 0
    for nodes in [(n0, n1), (n2, n3)]:
        geom = RowGeometry(
            sol_dict[nodes[0].geo.Rmid][0],
            sol_dict[nodes[0].geo.Rmid][0],
            sol_dict[nodes[0].geo.Height][0],
            sol_dict[nodes[1].geo.Height][0],
            sol_dict[nodes[0].geo.MeridionalAngle][0],
            sol_dict[nodes[1].geo.MeridionalAngle][0],
            sol_dict[nodes[1].geo.ChordAx][0],
            axial_offset=offset,
            force_straight=True,
        )
        plot_camberline(
            sol_dict[nodes[0].geo.MetalAngle],
            sol_dict[nodes[1].geo.MetalAngle],
            sol_dict[nodes[1].geo.ChordAx],
            ax=ax_cbl,
            color='k',
            axial_offset=offset,
        )
        opt_pitch = (
            2
            * np.pi
            * sol_dict[nodes[1].geo.Rmid]
            / sol_dict[nodes[1].geo.NumBladesOpt]
        )
        plot_camberline(
            sol_dict[nodes[0].geo.MetalAngle],
            sol_dict[nodes[1].geo.MetalAngle],
            sol_dict[nodes[1].geo.ChordAx],
            ax=ax_cbl,
            color='k',
            axial_offset=offset,
            tangential_offset=opt_pitch,
        )
        geom.plot_meridional_profile(ax=ax_mer, color='k')
        # Add offset
        offset += sol_dict[nodes[1].geo.ChordAx][0]

    # *** Camber lines

    plt.tight_layout()
    plt.show()
