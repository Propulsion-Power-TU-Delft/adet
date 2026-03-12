# === IMPORTS
from copy import deepcopy
import logging
from typing import Type

import CoolProp as cp
import matplotlib.pyplot as plt
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, Inlet, Shaft
from adet.components import ComponentNetwork
from adet.components.blade_row import DownstreamMixer
from adet.components.blade_row import plot_from_nodes
from adet.equations.base_equation import EquationBase, LossApplier
from adet.equations.definitions import (
    BoundaryLayerRatios,
    ClearanceByHeight,
    IsentropicProperties,
    RepeatedStage,
)
from adet.equations.fundamental import (
    BladeBlockage,
    FreeVortexDistribution,
    ZeroBlockage,
)
from adet.equations.geometrical import (
    MeridionalVariable,
    MinimalCamberLine,
    ModifiedZweifel,
    ParabolicCamberline,
)
from adet.equations.nondimensional import (
    FlowCoefficient,
    StaticTotalDegreeOfReaction,
    TotalStaticLoadingCoefficient,
    TotalTotalExpansionEfficiency,
    VolumetricFlowRatio,
    WorkCoefficient,
)
from adet.fluid.settings import ExternalFluidModel
from adet.fluid.settings import FluidSettings
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.losses.leakage import DentonLeakageLoss
from adet.losses.profile import DentonProfileLoss
from adet.losses.secondary import SecondaryBSM
from adet.registries import DefaultUnitsRegistry, GuessRegistry, VariableBoundsRegistry
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.iter import grouper
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)

setup_logger(
    logger,
    logging.DEBUG,
    logging.INFO,
    banned_keywords=['STREAM', 'findfont', 'sBIT'],
)
plt.close('all')

# === SETTINGS
NUM_SPAN = 1
SCALED = True
PLOTS = True
PRINTS = True
INITIAL_LOSS = PercentageEntropyLoss(0.0)

# ================================================

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('REFPROP', 'MM')
abs_state.debug_print = False

real_model = ExternalFluidModel(abs_state)
INLET_PRESSURE = 1.3 * abs_state.p_critical()
INLET_TEMPERATURE = 1.045 * abs_state.T_critical()
abs_state.update(cp.PT_INPUTS, INLET_PRESSURE, INLET_TEMPERATURE)


fluid_settings = FluidSettings(
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
        'Y_.*': 'dimensionless',
    }
)
_guess_reg = GuessRegistry()
_guess_reg.reset()
_guess_reg.from_dict(
    {
        'hdropCoeff': -0.8,
        'workCoeff': -0.8,
        'p_choke': 0.4 * INLET_PRESSURE,
        'reactDegree_ts': 0.5,
        'p': abs_state.p(),
        'T': abs_state.T(),
        'hmass': abs_state.hmass(),
        'smass': abs_state.smass(),
        'rhomass': abs_state.rhomass(),
        'k_prof': 0.3,  # Profile loading
        'zweifelCoeff': 0.85,
    }
)

# ================================================
# *** Variable bounds
_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
# _bounds_reg.ignore_defaults = True
_bounds_reg.from_dict(
    {
        # 'hdropCoeff': (-8.0, -0.2),
        'U': (0.0, 200.0),  # Reduce the search area
        'Vm': (20.0, 150.0),  # Reduce the search area
        'dev_angle': (-0.3, 0.3),
    }
)
if fluid_settings.model == real_model:
    _bounds_reg.from_dict(
        {
            'p': (abs_state.p_critical() * 0.5, 1.5 * INLET_PRESSURE),
            'T': (abs_state.T_critical() * 0.5, 1.5 * INLET_TEMPERATURE),
            'hmass': (abs_state.hmass() - 200**2, abs_state.hmass() + 200**2),
        }
    )

# ================================================
# *** Shafts
casing = Shaft(0.0, is_constrained=True)
shaft = Shaft(-1, is_constrained=False)

# ================================================
# *** Extra equations - Added after the first step


class LossMatcher(LossApplier):
    scaling_factor = (0.01,)

    def __init__(
        self,
        tip_gap: bool,
        scaling_factor: list[float] | None = None,
    ):
        super().__init__(scaling_factor)
        self.tip_gap = tip_gap

    def residual(
        self,
        stc_smass0,
        stc_smass1,
        oth_delta_smass_leakage1,
        oth_delta_smass_profile1,
        oth_delta_smass_secondary1,
    ):
        if self.tip_gap:
            return stc_smass1 - (
                stc_smass0
                + oth_delta_smass_leakage1
                + oth_delta_smass_profile1
                + oth_delta_smass_secondary1
            )
        return stc_smass1 - (
            stc_smass0 + oth_delta_smass_profile1 + oth_delta_smass_secondary1
        )


DUTY_COEFFS = {
    'oth_flowCoeff1': 0.55,
    'oth_volflowRatio1': 4,
    'oth_reactDegree_ts1': 0.3,
    'oth_ts_loadCoeff1': 4.0,
}

MIXING_EQS: dict[
    Type[EquationBase],
    int | tuple[int, ...],
] = {
    # Blockage of blade + b.l.
    BladeBlockage: 1,
    BoundaryLayerRatios: 1,
    # SieverdingBasePressure: (0, 1),
    ModifiedZweifel: (0, 1),
}

LOSS_MODELS: dict[
    Type[EquationBase],
    int | tuple[int, ...],
] = {
    # *** Blade row losses
    IsentropicProperties: (0, 1),
    SecondaryBSM: (0, 1),
    DentonProfileLoss: (0, 1),
    DentonLeakageLoss: (0, 1),
    ClearanceByHeight: 1,
}

# ================================================
# *** Inlet conditions
inlet = Inlet(
    {
        'oth': {
            'cum_massflow': 1,
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'hubtipRatio': 0.81,
        },
        'tot': {
            'p': abs_state.p(),
            'hmass': abs_state.hmass(),
        },
    }
)


stator = BladeRow(
    name='Stator',
    shaft=casing,
    row_type='stator',
    in_constraints={
        'geo': {
            'thick_by_pitch': 0.04,
        },
    },
    out_constraints={
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'thick_by_pitch': 0.02,
            'clearance_by_height': 0.01,
            # *** Num blades
            'num_blades': 20,
        },
        'oth': {  # NOTE: These are not used on first pass
            # *** Boundary layer ratios
            'mom_by_bld': 0.075,
            'disp_by_mom': 2,
            'disp_by_hgt': 0.05,  # endwall
            # *** Profile loss coeff
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,
            'xi_by_camb_len_B': 0.675,
            # *** Tip leakage discharge coeff
            'dischCoeff': 0.35,
        },
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # No deviation (accounted in mixers)
        MinimalCamberLine(): (0, 1),
        INITIAL_LOSS: (0, 1),
    },
    constant_variables=['geo_rr_midspan'],
)

# ============ Modify rows
rotor = deepcopy(stator)  # Reuse the stator as template
rotor.shaft = shaft  # Assign the rotating shaft
rotor.name = 'rotor'  # Not strictly required
rotor.row_type = 'rotor'  # Set the type
rotor.add_equation(WorkCoefficient(), (0, 1))
rotor.set_boundary_cond('geo_aspRatio1', 3.0)
stator.set_boundary_cond('geo_flare_angle1', Quantity(30, 'deg'))
# *** Duty coefficients
rotor.bc_from_dict(DUTY_COEFFS)  # Duty coefficients at node 1

# ================================================
# Create network
ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(num_span=NUM_SPAN),
    components=[stator, rotor],
)

rotor.set_spanwise_constant('geo_chord_ax1')
stator.set_spanwise_constant('geo_hh0', 'geo_chord_ax1')
rotor.copy_from_previous('geo_hh', 'geo_rr')
rotor.remove_equation(MeridionalVariable, 0)

if NUM_SPAN > 1:
    # Free vortex at stator and rotor outlets
    # TODO: Impose on mixers directly
    rotor.add_equation(FreeVortexDistribution(), 1)
    stator.add_equation(FreeVortexDistribution(), 1)

# Repeated stage definition
ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
# Inlet-to-outlet equations
ntw.system.add_equation(FlowCoefficient(), (0, 3))
ntw.system.add_equation(VolumetricFlowRatio(), (0, 3))
ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, 3))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))

# Build
ntw.system.build(SCALED)

# ============ Isentropic Solution ============
rootfinder_is = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.hessian_approximation': 'limited-memory',
    },
)

rtfn_kinsol = ntw.system.make_rootfinder('kinsol')

x0_is = ntw.system.get_scaled_guess()
kn_is = ntw.system.get_scaled_constraints()
bnd_is = ntw.system.get_arguments_bounds({'kin_alpha0': (-0.7, 0.7)})
solution = solve_root_problem(
    rootfinder_is,
    x0_is,
    kn_is,
    bnd_is,
    suppress_output=False,
)
solution = solve_root_problem(rtfn_kinsol, solution, kn_is)

stator_is_equations = stator._equations.copy()
rotor_is_equations = rotor._equations.copy()

# Write solution to dict for reading for next solution
sol_dict_is = ntw.system.write_solution_to_nodes(solution)

# ========================== MIXERS
user = input('INPUT >>> Continue with mixers? [y/n] ')
if user in ('y', 'Y'):
    # Create mixer objects
    sta_mixer = DownstreamMixer('sta_mixer')
    rot_mixer = DownstreamMixer('rot_mixer')
    rot_mixer.bc_from_dict(DUTY_COEFFS)

    # Remove number of blades and use zweifel
    rotor.rm_boundary_cond('geo_num_blades1')
    stator.rm_boundary_cond('geo_num_blades1')
    rotor.set_boundary_cond('geo_zweifelCoeff1', 0.85)
    stator.set_boundary_cond('geo_zweifelCoeff1', 0.85)

    ntw = ComponentNetwork(
        ntw.system.fluid_settings,
        inlet,
        CasadiSystem(NUM_SPAN),
        [
            stator,
            sta_mixer,
            rotor,
            rot_mixer,
        ],
    )
    # *** Node indices
    sta_in = 0
    sta_out = 1
    # Intermediate mixer out
    mix_out = 2 * ntw.components.index(sta_mixer) + 1
    rot_in = 2 * ntw.components.index(rotor)
    rot_out = rot_in + 1
    fin_node = 2 * len(ntw.components) - 1

    # *** Transpose dictionaries
    mixing_guess_dict = {}
    if isinstance(ntw.components[1], DownstreamMixer):
        mixing_guess_dict.update(
            **{k: v for k, v in sol_dict_is.items() if k.endswith(('0', '1'))}
        )
        # 1. Copy from stator outlet to mixer
        mixing_guess_dict.update(
            **{k.replace('1', '2'): v for k, v in sol_dict_is.items()}
        )
        mixing_guess_dict.update(
            **{k.replace('1', '3'): v for k, v in sol_dict_is.items()}
        )
        # 2. Shift rotor guesses
        mixing_guess_dict.update(
            **{k.replace('2', '4'): v for k, v in sol_dict_is.items()}
        )
        mixing_guess_dict.update(
            **{k.replace('3', '5'): v for k, v in sol_dict_is.items()}
        )
    if isinstance(ntw.components[-1], DownstreamMixer):
        # 1. Copy from isentropic rotor outlet to mixer
        mixing_guess_dict.update(
            **{k.replace('3', '6'): v for k, v in sol_dict_is.items()}
        )
        mixing_guess_dict.update(
            **{k.replace('3', '7'): v for k, v in sol_dict_is.items()}
        )

    STAGE_POSITIONS = (0, mix_out, rot_in, fin_node)
    # *** Re-add equations in correct position
    ntw.system.add_equation(RepeatedStage(), STAGE_POSITIONS)
    ntw.system.add_equation(StaticTotalDegreeOfReaction(), STAGE_POSITIONS)

    ntw.system.add_equation(FlowCoefficient(), (0, fin_node))
    ntw.system.add_equation(VolumetricFlowRatio(), (0, fin_node))
    ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, fin_node))
    ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, fin_node))

    rotor.remove_equation(ZeroBlockage, 1)
    stator.remove_equation(ZeroBlockage, 1)
    for eq, pos in MIXING_EQS.items():
        stator.add_equation(eq(), pos)
        rotor.add_equation(eq(), pos)

    ntw.build()

    x0_mixing = ntw.system.get_scaled_guess(mixing_guess_dict)
    kn_mixing = ntw.system.get_scaled_constraints()
    bnd_mixing = ntw.system.get_arguments_bounds()

    err_on_fail = int(
        input('INPUT >>> Fail on rootfinding error? [0/1] '),
    )
    rootfinder_mixing = ntw.system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': bool(err_on_fail),
            'ipopt.hessian_approximation': 'limited-memory',
        },
    )

    solution = solve_root_problem(
        rootfinder_mixing,
        x0_mixing,
        kn_mixing,
        bnd_mixing,
        suppress_output=False,
        perturbate_guess=False,
    )

    rtfn_kinsol = ntw.system.make_rootfinder(
        'kinsol',
        opts={'error_on_fail': bool(err_on_fail)},
    )
    mixing_sol_dict = ntw.system.write_solution_to_nodes(solution)


# ========================== LOSSES
user = input('INPUT >>> Continue with losses? [y/n] ')
if user in ('y', 'Y'):
    # --- Remove the first computation loss
    rotor.remove_equation(INITIAL_LOSS.__class__, (0, 1))
    stator.remove_equation(INITIAL_LOSS.__class__, (0, 1))
    # --- Add loss applier function
    stator.add_equation(LossMatcher(tip_gap=False), (0, 1))
    rotor.add_equation(LossMatcher(tip_gap=True), (0, 1))

    for eq, pos in LOSS_MODELS.items():
        stator.add_equation(eq(), pos)
        rotor.add_equation(eq(), pos)

    ntw.build()

    x0_loss = ntw.system.get_scaled_guess(mixing_sol_dict)
    kn_loss = ntw.system.get_scaled_constraints()
    bnd_loss = ntw.system.get_arguments_bounds()

    err_on_fail = int(
        input('INPUT >>> Fail on rootfinding error? [0/1] '),
    )
    rootfinder_loss = ntw.system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': bool(err_on_fail),
            'ipopt.hessian_approximation': 'limited-memory',
        },
    )

    solution = solve_root_problem(
        rootfinder_loss,
        x0_loss,
        kn_loss,
        bnd_loss,
        suppress_output=False,
        perturbate_guess=False,
    )

    sol_dict_loss = ntw.system.write_solution_to_nodes(solution)


ntw.print_structure()


# ------------ PLOTS ---------------------
if PLOTS:
    FONTSIZE = 18
    FONTDICT = {'fontsize': FONTSIZE}

    # Plot velocity triangles for each node
    for i, node in enumerate(ntw.system.nodes):
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        node.kin.plot(node.geo, FONTSIZE, ax)
        plt.title(f'Node number {i}')

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
    ax_merid.set_ylim(-0.01 * max_Y, max_Y)
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
        if not isinstance(comp, BladeRow):
            continue
        idx_map = comp.network_maps[ntw]
        inl_node = ntw.system.nodes[idx_map[0]]
        out_node = ntw.system.nodes[idx_map[1]]
        ax_chord = out_node.geo.chord_ax[0]

        # Plot meridional profile
        is_stator = comp.row_type == 'stator'
        color = 'steelblue' if is_stator else 'coral'

        lines = plot_from_nodes(
            inl_node,
            out_node,
            False,
            offset,
            ax=ax_merid,
            color=color,
        )
        ax_merid.plot(NUM_SPAN * [offset], inl_node.geo.rr, 'o', color='r')
        ax_merid.plot(NUM_SPAN * [offset] + ax_chord, out_node.geo.rr, 'o', color='r')

        ax_merid.plot(
            NUM_SPAN * [offset],
            inl_node.geo.rr + inl_node.geo.hh / 2,
            '_',
            color='g',
        )
        ax_merid.plot(
            NUM_SPAN * [offset],
            inl_node.geo.rr - inl_node.geo.hh / 2,
            '_',
            color='b',
        )

        ax_merid.plot(
            NUM_SPAN * [offset] + ax_chord,
            out_node.geo.rr + out_node.geo.hh / 2,
            '_',
            color='g',
        )
        ax_merid.plot(
            NUM_SPAN * [offset] + ax_chord,
            out_node.geo.rr - out_node.geo.hh / 2,
            '_',
            color='b',
        )

        # Plot camberlines at midspan (3 blades for all rows)
        midspan_idx = ntw.system.num_span // 2
        inlet_angle = inl_node.geo.metal_angle[midspan_idx]  # pyright:ignore
        outlet_angle = out_node.geo.metal_angle[midspan_idx]  # pyright:ignore
        chord_ax = out_node.geo.chord_ax[midspan_idx]  # pyright:ignore
        pitch = out_node.geo.pitch[midspan_idx]  # pyright:ignore

        num_blades = 3

        for blade_num in range(num_blades):
            pbl.plot_camber_line(
                ax_camber,
                inlet_angle,
                outlet_angle,
                chord_ax,
                'k',
                axial_offset=offset,
                tangential_offset=blade_num * pitch,
            )
        # Plot hub and tip camberlines
        pbl.plot_camber_line(
            ax_camber,
            inl_node.geo.metal_angle[0],
            out_node.geo.metal_angle[0],
            out_node.geo.chord_ax[0],
            'orange',
            axial_offset=offset,
        )
        pbl.plot_camber_line(
            ax_camber,
            inl_node.geo.metal_angle[-1],
            out_node.geo.metal_angle[-1],
            out_node.geo.chord_ax[-1],
            'seagreen',
            axial_offset=offset,
        )

        offset += ax_chord * 1.1

    # Add axis reference line for meridional plot
    ax_merid.plot(
        [0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2.5
    )

    plt.tight_layout()

    blade_rows = list(grouper(num_nodes, 2, incomplete='ignore'))

    fig, ax = plt.subplots(figsize=(5, 5))

    cmap = plt.colormaps.get('autumn')
    for idx, (n0_idx, n1_idx) in enumerate(blade_rows):
        inl_node = ntw.system.nodes[n0_idx]
        out_node = ntw.system.nodes[n1_idx]

        smass_in = inl_node.stc.smass
        smass_out = out_node.stc.smass

        # Extract blade height for normalization
        height_in = inl_node.geo.height
        # Normalize radial positions by blade height (hub = 0, tip = 1)
        radii = inl_node.geo.rr
        rr_midspan = inl_node.geo.rr_midspan
        span_normalized = (radii - (rr_midspan - height_in / 2)) / height_in

        # Determine blade type and color
        is_stator = (n0_idx // 2) % 2 == 0
        blade_type = 'Stator' if is_stator else 'Rotor'
        stage_num = n0_idx // 4

        color = cmap(idx / (len(ntw.components) - 0.8))  # pyright:ignore

        if idx == 0:
            ax.plot(
                span_normalized,
                smass_in,
                label='Inlet',
                color='blue',
                linestyle='-',
                linewidth=2,
                marker='o',
            )

        ax.plot(
            span_normalized,
            smass_out,
            label=f'Stage {stage_num} {blade_type}',
            color=color,
            linestyle='-',
            linewidth=2,
            marker='o',
        )

    ax.set_xlabel('Normalized Span (hub=0, tip=1)', FONTDICT)
    ax.set_title('Entropy [J / kg / K]', FONTDICT)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=FONTSIZE // 1.5)
    fig.tight_layout()

# ________________________________________________
# _________________ PRINTS _______________________

print(f'Num updates = {real_model.eos_object.num_updates}')

nodes = {}
for i, node in enumerate(ntw.system.nodes):
    # For simpler access set n0, n1, n2, ...
    nodes[i] = node

# A little staircase :)
n0 = nodes[0]
n1 = nodes[1]
if ntw.num_components > 1:
    n2 = nodes[2]
    n3 = nodes[3]
    if ntw.num_components > 2:
        n4 = nodes[4]
        n5 = nodes[5]
        if ntw.num_components > 3:
            n6 = nodes[6]
            n7 = nodes[7]


answer = input('INPUT >>> Show plots? [y/n]')
if answer in ('Y', 'y'):
    plt.show(block=False)
    input('Press enter to close ')
plt.close('all')

print(f'Inlet mach is {n0.kin.mach}')
