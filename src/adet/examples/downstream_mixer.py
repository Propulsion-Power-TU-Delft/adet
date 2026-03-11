# === IMPORTS
# Standard library
import logging

import matplotlib.pyplot as plt
import numpy as np
import CoolProp as cp
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, Inlet, Shaft
from adet.components import ComponentNetwork
from adet.components.blade_row import DownstreamMixer
from adet.components.blade_row import plot_from_nodes
from adet.diagnostics import SystemDiagnostics
from adet.equations.definitions import BoundaryLayerRatios, IsentropicProperties
from adet.equations.fundamental import BladeBlockage, ChokingCriterion
from adet.equations.geometrical import MinimalCamberLine, ParabolicCamberline
from adet.equations.nondimensional import (
    StaticPressRatio,
    TotalTotalExpansionEfficiency,
)
from adet.equations.utils import residual_debugger
from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel
from adet.fluid.settings import FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.losses.mixing import (
    AungierDeviationModel,
    SieverdingBasePressure,
)
from adet.registries import DefaultUnitsRegistry, GuessRegistry, VariableBoundsRegistry
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)

setup_logger(
    logger,
    logging.INFO,
    logging.INFO,
    banned_keywords=['STREAM', 'findfont', 'sBIT'],
)

# === SETTINGS
NUM_SPAN = 1
SCALED = True
PLOTS = True
PRINTS = True
INITIAL_LOSS = PercentageEntropyLoss(0.0)

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('REFPROP', 'NOVEC649')
abs_state.debug_print = False
id_state = IdealGasState(1.4, 287, 1.8e5)

real_model = ExternalFluidModel(abs_state)
ideal_model = AnalyticalFluidModel(id_state)

settings = FluidSettings(
    model=real_model,
    update_variables=('p', 'hmass', 'T'),
    update_length=2,
)

_defreg = DefaultUnitsRegistry()
_defreg.from_dict(
    {
        'xi_by_camb_.*': 'dimensionless',
        'Cd_prof': 'dimensionless',
        'k_prof': 'dimensionless',
    }
)
# Variable guesses
_guess_reg = GuessRegistry()
_guess_reg.reset()
_guess_reg.from_dict(
    {
        'pRatio': 0.9,
        'p_choke': 3e5,
        'VmRatio': 1.3,
        'Vm': 30,
        'Wm': 30,
        'Vt': 10,
        'Wt': 10,
    }
)

# Variable bounds
_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
_bounds_reg.ignore_defaults = False
_bounds_reg.from_dict(
    {
        'delta_smass_mixing': (0.0, 10.0),
    }
)

INLET_PRESSURE = 2.3e5
INLET_TEMPERATURE = 383
abs_state.update(cp.PT_INPUTS, INLET_PRESSURE, INLET_TEMPERATURE)
_bounds_reg.from_dict(
    {
        'p': (0.6 * INLET_PRESSURE, 1.5 * INLET_PRESSURE),
        'T': (0.5 * INLET_PRESSURE, 1.5 * INLET_TEMPERATURE),
        'hmass': (abs_state.hmass() - 200**2, 1.2 * abs_state.hmass()),
    }
)

# *** Shafts
shaft = Shaft(
    omega=Quantity(0.0, 'rpm'),
    is_constrained=True,
)

# COMPONENT STACK
inlet = Inlet(
    {
        'kin': {
            'beta': 0.0,
            # 'mach': 0.15,
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': 0.1,
            'hubtipRatio': 0.7,
        },
        'tot': {
            'p': INLET_PRESSURE,
            'hmass': abs_state.hmass(),
        },
    }
)

# row = Models inlet-to-throat here
row = BladeRow(
    name='stator',
    shaft=shaft,
    row_type='rotor',
    in_constraints={
        'geo': {
            'thick_by_pitch': 0.04,
        },
    },
    out_constraints={
        'kin': {
            # 'mach': 0.4,
        },
        'geo': {
            'metal_angle': Quantity(50, 'deg'),
            # Meridional
            'meridional_angle': Quantity(0, 'deg'),
            # Blade
            'aspRatio': 1.8,
            # 'chord_ax': 0.1,
            'num_blades': 40,
            # 'solidity': 1.0,
            'thick_by_pitch': 0.01,
            'heightRatio': 1.0,
        },
        'oth': {
            'mom_by_bld': 0.075,
            'disp_by_mom': 2,
            'disp_by_hgt': 0.05,  # Used by sec losses
            # Profile losses
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,
            'xi_by_camb_len_B': 0.675,
        },
    },
    extra_equations={
        # Camberline model
        MinimalCamberLine(): (0, 1),
        # ParabolicCamberline(): (0, 1),
        # |> Losses & Dev
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        # |> Boundary layer properties
        BladeBlockage(): 1,
        BoundaryLayerRatios(): 1,
        SieverdingBasePressure(): (0, 1),
        INITIAL_LOSS: (0, 1),
    },
    constant_variables=['geo_rr_midspan'],
)

mixer = DownstreamMixer(
    'twitch',
    outlet_bc={
        # 'oth': {'pRatio': 0.7},  # Mixer in-to-out
        'kin': {'mach': 0.2},  # Mixed-out mach
    },
    extra_equations={
        StaticPressRatio(): (0, 1),
        AungierDeviationModel(): 1,
    },
)

# Create network
ntw = ComponentNetwork(
    settings,  # Fluid settings
    inlet,  # Inlet conditions
    CasadiSystem(num_span=NUM_SPAN),  # Backend
    [
        row,
        mixer,
    ],
)

row.set_spanwise_constant('geo_hh0', 'kin_Vm1')
row.set_spanwise_constant('geo_chord_ax1')


if ntw.num_components == 2 and shaft.omega > 0:
    ntw.system.add_equation(IsentropicProperties(), (0, 3))
    # WARN: For a stator this crashes the code
    ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))


ntw.system.build(SCALED)

rootfinder = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.max_wall_time': 10,
    },
)
x0 = ntw.system.get_scaled_guess()
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds(
    {
        'kin_mach0': (0.0, 0.9),
    }
)

# diag = SystemDiagnostics(ntw.system, kn)

solution = solve_root_problem(rootfinder, x0, kn, bnd)

rtfn = ntw.system.make_rootfinder('kinsol')
solution = solve_root_problem(rtfn, solution, kn)

sol_dict = ntw.system.write_solution_to_nodes(solution)
ntw.print_structure()


nodes = {}
for i, node in enumerate(ntw.system.nodes):
    # For simpler access set n0, n1, n2, ...
    nodes[i] = node
n0 = nodes[0]
n1 = nodes[1]

if mixer in ntw.components:
    ntw.system.nodes[-1].geo.add_variable('chord_ax', 0.2 * n1.geo.chord_ax)


if PLOTS:
    FONTSIZE = 18
    FONTDICT = {'fontsize': FONTSIZE}

    # Plot velocity triangles
    for i, node in enumerate(ntw.system.nodes):
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        node.kin.plot(node.geo, FONTSIZE, ax)
        plt.title(f'Node number {i}')

    # Plot entropy rise
    fig, ax = plt.subplots()
    ax.set_title('Entropy rise')
    smass0 = ntw.system.nodes[0].stc.smass
    for i, node in enumerate(ntw.system.nodes):
        # Plot entropy distributions
        ax.plot(node.geo.rr, node.stc.smass - smass0, label=f'Node {i}')  # pyright:ignore
        ax.legend()

    plt.tick_params(labelsize=FONTSIZE / 1.5 // 1)

    num_nodes = range(len(ntw.system.nodes))

    # Plot meridional profile and camberlines in subplots
    fig, (ax_merid, ax_camber) = plt.subplots(1, 2, figsize=(14, 6))

    # Configure meridional profile subplot
    ax_merid.axis('equal')
    ax_merid.set_ylabel('radius [m]', {'fontsize': 18})
    ax_merid.set_xlabel('axial  [m]', {'fontsize': 18})
    max_Y = (
        1.1
        * (
            ntw.system.nodes[-1].geo.get('rr_midspan').magnitude
            + ntw.system.nodes[-1].geo.get('height').magnitude / 2
        )[0]
    )
    ax_merid.set_ylim(-0.01, max_Y)
    ax_merid.tick_params('both', labelsize=18)
    ax_merid.grid()
    ax_merid.set_title('Meridional profile', {'fontsize': 18})

    # Configure camberline subplot
    ax_camber.set_title('Camberlines at midspan', {'fontsize': 18})
    ax_camber.axis('equal')
    ax_camber.tick_params('both', labelsize=18)
    ax_camber.grid()

    # Merged loop for both plots
    pbl = ParabolicCamberline()
    offset = 0.0
    for comp in ntw.components:
        idx_map = comp.network_maps[ntw]
        inl_node = ntw.system.nodes[idx_map[0]]
        out_node = ntw.system.nodes[idx_map[1]]
        ax_chord = out_node.geo.chord_ax[0]

        # Plot meridional profile
        lines = plot_from_nodes(
            inl_node,
            out_node,
            False,
            offset,
            ax=ax_merid,
        )

        # Plot camberlines at midspan (3 blades for rotor, 1 for stator)
        midspan_idx = ntw.system.num_span // 2
        inlet_angle = inl_node.geo.metal_angle[midspan_idx]  # pyright:ignore
        outlet_angle = out_node.geo.metal_angle[midspan_idx]  # pyright:ignore
        pitch = inl_node.geo.pitch[midspan_idx]  # pyright:ignore
        num_plt_blades = 3  # blades to plot

        for blade_num in range(num_plt_blades):
            pbl.plot_camber_line(
                ax_camber,
                inlet_angle,
                outlet_angle,
                ax_chord,
                'k',
                axial_offset=offset,
                tangential_offset=blade_num * pitch,
            )

        offset += ax_chord * 1.1

    # Add axis reference line for meridional plot
    ax_merid.plot(
        [0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2.5
    )

    plt.tight_layout()

print(f'Num updates = {real_model.eos_object.num_updates}')
print(f'Out row mach {n1.kin.mach}')

if ntw.num_components > 1:
    n2 = nodes[2]
    n3 = nodes[3]
    print(f'Entropy rise {n3.stc.smass - n2.stc.smass}')
    print(f'Deviation {np.sign(n2.geo.metal_angle) * (n2.kin.beta - n3.kin.beta)}')

    q = 0.5 * n2.stc.rhomass * n2.kin.W**2
    w = n2.geo.pitch * np.cos(n2.geo.metal_angle)
    cpb = (n2.oth.p_base - n2.stc.p) / q

    zeta_inc = (
        -(cpb * n2.geo.bld_thick) / w
        + 2 * n2.oth.mom_thick / w
        + ((n2.oth.disp_thick + n2.geo.bld_thick) / w) ** 2
    )

    zeta_actual = (n2.rlt.p - n3.rlt.p) / q

    print(f'Incompressible vs actual zeta {zeta_inc}, {zeta_actual}')

user = input('>>> INPUT: Show plots [y/n] ')
if user in ('y', 'Y'):
    plt.show(block=False)
    input('Enter to close')
plt.close('all')

globals().update(residual_debugger(AungierDeviationModel(), [n1]))
RUN_SWEEP = True
N_PTS = 100
if RUN_SWEEP:
    mach_out_idx = ntw.system.constraints.index('kin_mach3')
    rtfn = ntw.system.make_rootfinder('kinsol')
    out_machs = np.linspace(0.2, 0.9, N_PTS)
    loss_coeffs = []
    deviations = []
    for m in out_machs:
        flag_error = 0
        kn[mach_out_idx] = np.array([m]) * ntw.system.constraints_scaling[mach_out_idx]
        x0 = ntw.system.get_scaled_guess(sol_dict)
        try:
            solution = solve_root_problem(
                rtfn,
                x0,
                kn,
                bnd,
                suppress_output=False,
            )
        except RuntimeError:
            flag_error = 1

        if flag_error:
            y_loss = np.array([np.nan])
            dev = np.array([np.nan])
        else:
            y_loss = (n2.rlt.p - n3.rlt.p) / (n2.rlt.p - n2.stc.p)
            dev = n2.kin.beta - n3.kin.beta

        loss_coeffs.append(y_loss)
        deviations.append(dev)

        sol_dict = ntw.system.write_solution_to_nodes(solution)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(out_machs, loss_coeffs, label='zeta')
    ax[1].plot(out_machs, np.array(deviations) * 180 / np.pi, label='deviations [deg]')
    ax[0].legend()
    ax[1].legend()

plt.show(block=True)
