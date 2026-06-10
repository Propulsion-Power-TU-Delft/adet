# === IMPORTS
from adet.equations.utils import residual_debugger
from adet.components.blade_row import RowGeometry
from adet.tools.plotting import plot_velocity_triangles, setup_mpl, plot_camberline
import logging
import matplotlib.pyplot as plt
from adet.tools.loggers import setup_logger
from copy import deepcopy

from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.definitions import RepeatedStage
from adet.equations.geometrical import MeridionalGeometry, ModifiedZweifel
from adet.equations.nondimensional import (
    FlowCoefficient,
    StaticTotalDegreeOfReaction,
    TotalTotalExpansionEfficiency,
    WorkCoefficient,
)
from adet.fluid.settings import FluidModel, FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
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
        # n0.kin.V_mer: 40,
        # *** Inlet geometry
        n0.geo.Rmid: 0.1,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        # *** Inlet total conditions
        n0.tot.Pressure: 10e5,
        n0.tot.Temperature: 500,
    }
)


casing = Shaft(0, is_constrained=True)
shaft = Shaft(0, is_constrained=False)


stator = BladeRow(
    name='stator',
    shaft=casing,
    bound_cond={
        n0.geo.ThickByPitch: 0.04,
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.geo.ThickByPitch: 0.02,
        # n1.geo.AspectRatio: 3.0,
        n1.geo.FlareAngle: Quantity(30, 'deg'),
        # n1.geo.ClearanceByHeight: 0.01,
        n1.geo.NumBlades: 10,
        # n1.geo.ZweifelCoeff: 0.85,
        # *** Boundary layer
        n1.oth.MomByBld: 0.075,
        n1.oth.DispByMom: 2,
        n1.oth.DispByHgt: 0.05,
        # *** Losses
        n1.oth.CdProfile: 0.002,
        n1.oth.XiCambLenA: 0.375,
        n1.oth.XiCambLenB: 0.675,
        n1.oth.DischCoeff: 0.35,
        # n1.stc.Pressure: 1.462617e6, # Legacy ?
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # No deviation
        IsentropicLink(): (0, 1),
        # ModifiedZweifel(): (0, 1),
    },
)
stator.set_component_constants(n0.geo.Rmid.Glob)

# ============ Modify rotor
rotor = deepcopy(stator)  # Reuse the stator as template
rotor.shaft = shaft  # Assign the rotating shaft
rotor.name = 'rotor'

fluid_model = FluidModel(abs_state)
fluid_settings = FluidSettings(
    fluid_model,
    (n0.stc.Pressure, n0.stc.Temperature),
)

ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(1),
    [stator, rotor],
)


ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
ntw.system.add_equation(FlowCoefficient(), (0, 3))
ntw.system.add_equation(WorkCoefficient(), (0, 3))
# ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))

rotor.set_spanwise_constant(n1.geo.ChordAx)
stator.set_spanwise_constant(
    # Uniform inlet
    n0.kin.V_mer,
    n0.geo.HDistr,
    # Uniform chords along the span
    n1.geo.ChordAx,
)
# Copy meridional geometry from previous node
# instead of computing it
rotor.copy_from_previous(n0.geo.HDistr, n0.geo.RDistr)
rotor.remove_equation(MeridionalGeometry, 0)

stator.set_bc_from_dict(
    {
        n1.kin.RelMach: 0.8,
    },
)

rotor.set_bc_from_dict(
    {
        n1.geo.HubTipRatio: 0.818,
        n1.ndim.FlowCoeff: 0.4,
        n1.ndim.WorkCoeff: -1.0,
        n1.ndim.DegreeOfReactionTS: 0.5,
    }
)
ntw.build()
input('Continue?')


x0 = ntw.system.get_scaled_guess(fallback=0.8)
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds(
    {
        n0.stc.Pressure.Glob: (100.0, 13e5),
        n0.stc.Temperature.Glob: (60.0, 500),
    },
    ignore_defaults=False,
)


# Ipopt
rtfn = ntw.system.make_rootfinder('ipopt', {'error_on_fail': False})
sol = solve_root_problem(rtfn, x0, kn, bnd)

# Kinsol
rtfn = ntw.system.make_rootfinder('kinsol')
sol = solve_root_problem(rtfn, sol, kn)

data = ntw.system.sol_to_dict(sol)

# globals().update(residual_debugger(ModifiedZweifel(), [0, 1], data))

if True:
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
        plot_camberline(
            data[nodes[0].geo.MetalAngle],
            data[nodes[1].geo.MetalAngle],
            data[nodes[1].geo.ChordAx],
            ax=ax_cbl,
            color='k',
            axial_offset=offset,
            tangential_offset=data[nodes[1].geo.Pitch],
        )
        geom.plot_meridional_profile(ax=ax_mer, color='k')
        # Add offset
        offset += data[nodes[1].geo.ChordAx][0]

    # *** Camber lines

    plt.tight_layout()
    plt.show()
