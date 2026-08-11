"""
Design of a generic centrifugal compressor for a fuel cell
air supply system design
"""

# Design problem:
# Tt1 = -22.5 degC
# pt1 = 0.45 barA
# pt2 = 2.025 barA
# m_flow = 1.5 kg/s

# shape factor k = 1 - (Rh1/Rs1)^2 = 0.9
# swallowing capacity phi_t1 = 0.05
# outlet flow angle alpha2 = 65 deg
# pressure recovery factor diffuser Cp = 0.5
# ratio axial length to impeller outlet radius Lax/R2 = 0.7

import logging

import matplotlib.pyplot as plt
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.components import BladeRow
from adet.components.blade_row import RowGeometry
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.definitions import EffectiveBladeNumber
from adet.equations.geometrical import ChordAxbyOutradius
from adet.equations.nondimensional import (
    SwallowingCapacity,
    TotalTotalCompressionEfficiency,
    WorkCoefficient,
)
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.losses.basic import (
    IsentropicLink,
    ZeroDeviation,
)
from adet.losses.compressors import (
    BackstromSlip,
    CaseyRushInletFunc,
    CompressorShapeFactor,
)
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_velocity_triangles
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

n0 = NodeVariables(0)
n1 = NodeVariables(1)

NUM_SPAN = 1

# +++ Shafts
shaft = Shaft(
    omega=-1,
    is_constrained=False,
)

casing = Shaft(
    omega=Quantity(0, 'rpm'),
    is_constrained=True,
)

# +++ Fluid settings
realgas_state = DebugAbstractState('HEOS', 'Air')
idealgas_state = IdealGasState(1.4, 287, 1.8e-5)

fluid_settings = FluidSettings(
    fluid_state=idealgas_state,
    update_variables=(n0.stc.Pressure, n0.stc.Temperature),
)


# +++ Boundary conditions
inlet = Inlet(
    boundary_conditions={
        n0.tot.Pressure: Quantity(0.45, 'bar'),
        n0.tot.Temperature: Quantity(-22.5, 'degC'),
        n0.kin.FlowAngleAbs: Quantity(0.0, 'deg'),
        n0.oth.TotMassFlow: 1.5,
    }
)

EQS_ISENTROPIC = {
    # Losses are computed but not added
    ZeroDeviation(): 1,
    IsentropicLink(): (0, 1),
}


EQS_WITH_LOSSES = {
    BackstromSlip(): (0, 1),
    # Losses are added
    IsentropicLink(): (0, 1),
}

# +++ Components
rotor = BladeRow(
    name='rotor',
    shaft=shaft,
    bound_cond={
        # *** Geometry
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        n0.geo.ThickByPitch: 0.02,
        n1.kin.FlowAngleAbs: Quantity(65, 'deg'),
        n1.geo.MeridionalAngle: Quantity(90, 'deg'),
        n1.geo.ThickByPitch: 0.02,
        n1.geo.NumBlades: 15,
        n1.geo.NumSplitters: 15,
        # *** Design parameters
        n0.geo.ShapeCoeff: 0.9,
        n0.ndim.SwallowingCap: 0.05,
        n1.ndim.WorkCoeff: 0.5,
        n1.ndim.ChAxOutRadRatio: 0.7,
        n1.tot.Pressure: Quantity(2.025, 'bar'),
        # *** Loss parameters
        n1.oth.BlLoadingCoeff: 0.75,
        n0.geo.TipClearance: 1e-4,
        n1.geo.AbsRoughness: Quantity(0.01, 'mm'),
        n1.oth.SlipFactCoeff: 5.0,
    },
    extra_equations={
        # *** Design parameters
        ChordAxbyOutradius(): 1,
        SwallowingCapacity(): (0, 1),
        WorkCoefficient(): (0, 1),
        CompressorShapeFactor(): 0,
        CaseyRushInletFunc(): (0, 1),
        # *** Geometry
        EffectiveBladeNumber(): 1,
        # *** Metal angle <-[link]-> Flow angle
        ZeroDeviation(): 0,  # Zero incidence
        # *** Enthalpy definitions
        TotalTotalCompressionEfficiency(): (0, 1),
        **EQS_ISENTROPIC,
    },
)


BOUNDS = {
    n0.kin.BladeSpeed.Glob: (0.0, 500),
    n0.kin.Omega.Glob: (0.0, 1e4),
    n0.kin.FlowAngleRel.Glob: (-1.5, 0.0),
    n0.kin.Beta_tip.Glob: (-1.5, 0.0),
    n0.geo.Rhub: (0.01, 10.0),
    #
    n0.stc.Pressure.Glob: (100, 1e8),
    n0.stc.Temperature.Glob: (10, 1e8),
}

ntw = ComponentNetwork(
    fluid_settings=fluid_settings,
    inlet=inlet,
    backend=CasadiSystem(num_span=NUM_SPAN),
    components=[rotor],
)


ntw.build()

x0_is = ntw.system.get_guess(fallback=0.6)
kn_is = ntw.system.get_boundary_conds()
bnd_is = ntw.system.get_bounds(BOUNDS, ignore_defaults=False)

# IPOPT is very robust, KINSOL faster but fails more easily
rootfinder_des_is = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.tol': 1e-6,
        'ipopt.max_iter': 300,
        'ipopt.max_wall_time': 30,
        # 'ipopt.print_level': 3,
    },
)

# Get the isentropic solution
solution_is = solve_root_problem(
    rootfinder_des_is,
    x0_is,
    kn_is,
    bnd_is,
    suppress_output=False,
)

sol_is_dict = ntw.system.sol_to_dict(solution_is)

rtfn = ntw.system.make_rootfinder('kinsol')
solution_is = solve_root_problem(rtfn, solution_is, kn_is)

sol_loss_dict = ntw.system.sol_to_dict(solution_is)


# Remove isentropic and add losses
for eq, pos in EQS_ISENTROPIC.items():
    rotor.remove_equation(eq.__class__, pos)
for eq, pos in EQS_WITH_LOSSES.items():
    rotor.add_equation(eq, pos)

ntw.build()  # Rebuild

# Get
x0_loss = ntw.system.get_guess(sol_is_dict, fallback=0.8)
bnd_loss = ntw.system.get_bounds(BOUNDS)
rootfinder_loss = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.tol': 1e-7,
        'ipopt.print_level': 3,
    },
)

solution_loss = solve_root_problem(
    rootfinder_loss,
    x0_loss,
    kn_is,
    suppress_output=False,
)
sol_loss_dict = ntw.system.sol_to_dict(solution_loss)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

node_pairs = [
    (n0, n1),
]
for _, (inlet_n, outlet_n) in enumerate(node_pairs):
    for node_idx, n in enumerate([inlet_n, outlet_n]):
        ax = axs[node_idx]
        ax.set_aspect('equal')

        plot_velocity_triangles(
            sol_loss_dict[n.kin.V_tan],
            sol_loss_dict[n.kin.V_mer],
            sol_loss_dict[n.kin.BladeSpeed],
            sol_loss_dict[n.geo.RDistr],
            ax,
        )

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_aspect('equal')

r_in = float(sol_loss_dict[n0.geo.Rmid][0])
r_out = float(sol_loss_dict[n1.geo.Rmid][0])
height_in = float(sol_loss_dict[n0.geo.Height][0])
height_out = float(sol_loss_dict[n1.geo.Height][0])
mer_angle_in = float(sol_loss_dict[n0.geo.MeridionalAngle][0])
mer_angle_out = float(sol_loss_dict[n1.geo.MeridionalAngle][0])
axial_chord = float(sol_loss_dict[n1.geo.ChordAx][0])

geom = RowGeometry(
    r_in=r_in,
    r_out=r_out,
    height_in=height_in,
    height_out=height_out,
    mer_angle_in=mer_angle_in,
    mer_angle_out=mer_angle_out,
    axial_chord=axial_chord,
)
geom.plot_meridional_profile(color='k', ax=ax)

ax.set_title('Meridional Profile')
ax.set_xlabel(r'$z$ / [m]')
ax.set_ylabel(r'$r$ / [m]')
ax.grid(True, alpha=0.3)

fig.tight_layout()

show_plots = input('Show plots? [y/N] ').strip().lower() == 'y'
if show_plots:
    plt.show()
else:
    plt.close('all')
