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
from adet.equations.fundamental import FreeVortexDistribution
from adet.equations.geometrical import (
    MeridionalVariable,
    MinimalCamberLine,
    ParabolicCamberline,
)
from adet.equations.nondimensional import (
    FlowCoefficient,
    StaticTotalDegreeOfReaction,
    TotalStaticLoadingCoefficient,
    TotalTotalExpansionEfficiency,
    WorkCoefficient,
)
from adet.fluid.settings import ExternalFluidModel
from adet.fluid.settings import FluidSettings
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.losses.leakage import DentonLeakageLoss
from adet.losses.mixing import SieverdingBasePressure
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
    logging.INFO,
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
INLET_PRESSURE = 2.071 * abs_state.p_critical()
INLET_TEMPERATURE = 1.052 * abs_state.T_critical()
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
        'reactDegree_ts': 0.5,
        'p': abs_state.p(),
        'T': abs_state.T(),
        'hmass': abs_state.hmass(),
        'smass': abs_state.smass(),
        'rhomass': abs_state.rhomass(),
        'k_prof': 0.3,  # Profile loading
    }
)

# ================================================
# *** Variable bounds
_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
_bounds_reg.from_dict(
    {
        # 'delta_smass_.*': (0.0, 100.0),
        'hdropCoeff': (-8.0, -0.4),
        # 'eta_tt': (0.7, 1.0),
        'U': (0.0, 200.0),  # Reduce the search area
    }
)
if fluid_settings.model == real_model:
    _bounds_reg.from_dict(
        {
            'p': (abs_state.p_critical(), INLET_PRESSURE),
            'T': (abs_state.T_critical(), INLET_TEMPERATURE),
            'hmass': (abs_state.hmass() - 2 * 60**2, 1.2 * abs_state.hmass()),
        }
    )

# ================================================
# *** Shafts
casing = Shaft(0.0, is_constrained=True)
shaft = Shaft(-1, is_constrained=False)

# ================================================
# *** Extra equations - Added after the first step


class LossMatcher(LossApplier):
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


EXTRA_EQUATIONS: dict[
    Type[EquationBase],
    int | tuple[int, ...],
] = {
    ClearanceByHeight: 1,
    IsentropicProperties: (0, 1),
    BoundaryLayerRatios: 1,
    # ReducedThermoQuantities: 0,
    SieverdingBasePressure: (0, 1),
    SecondaryBSM: (0, 1),
    DentonProfileLoss: (0, 1),
    DentonLeakageLoss: (0, 1),
    # BladeBlockage: 1,
}

# ================================================
# *** Inlet conditions
inlet = Inlet(
    {
        'oth': {
            'cum_massflow': 1,
        },
        'kin': {
            'mermach': 0.1,
            # 'Vm': 100,
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'hubtipRatio': 0.7,
        },
        'tot': {
            'p': abs_state.p(),
            # 'T': abs_state.T(),
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
            # ***  height <-> chord
            'aspRatio': 2,
            'num_blades': 25,
            'thick_by_pitch': 0.02,
            'clearance_by_height': 0.01,
            # 'solidity': 1.3, # WARN: Applies along span
        },
        'oth': {
            # *** Boundary layer ratios
            'mom_by_bld': 0.075,
            'disp_by_mom': 2,
            'disp_by_hgt': 0.05,
            # *** Profile loss coeff
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,
            'xi_by_camb_len_B': 0.675,
            # *** Tip leakage discharge coeff
            'dischCoeff': 0.35,
        },
    },
    extra_equations={
        # *** Camberline model
        MinimalCamberLine(): (0, 1),
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,
        INITIAL_LOSS: (0, 1),
    },
    constant_variables=['geo_rr_midspan'],
)


rotor = deepcopy(stator)  # Reuse the ratios from stator
rotor.shaft = shaft  # Assign the rotating shaft
rotor.row_type = 'rotor'
rotor._equations.update(
    {
        WorkCoefficient(): (0, 1),
        FlowCoefficient(): (0, 1),
    }
)


mixer = DownstreamMixer(
    'Mixer',
    # The axial chord is just for plotting it like a blade row
    out_constraints={'geo': {'chord_ax': 0.01}},
)

# ================================================
# Create network
ntw = ComponentNetwork(
    fluid_settings,  # Fluid settings
    inlet,  # Inlet conditions
    CasadiSystem(num_span=NUM_SPAN),  # Backend
    components=[stator, rotor],
)

ntw.system.boundary_conditions[3]['oth']['flowCoeff'] = 0.4
ntw.system.boundary_conditions[3]['oth']['reactDegree_ts'] = 0.3
ntw.system.boundary_conditions[3]['oth']['ts_loadCoeff'] = 3
# ntw.system.boundary_conditions[3]['oth']['workCoeff'] = -1.3

final_node = ntw.num_components * 2 - 1

ntw.system.add_spanwise_constants('geo_hh0', 'geo_chord_ax1', 'geo_chord_ax3')
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, final_node))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, final_node))
ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))

if NUM_SPAN > 1:
    # Free vortex at stator and rotor outlets
    ntw.system.add_equation(FreeVortexDistribution(), 1)
    ntw.system.add_equation(FreeVortexDistribution(), final_node)
    # Rotor inlet geometry is continuous with stator outlet (no MeridionalVariable)
    ntw.system.remove_equation(MeridionalVariable, 2)
    ntw.system.add_equalities(
        ('geo_hh1', 'geo_hh2'),
        ('geo_rr1', 'geo_rr2'),
    )

ntw.system.build(SCALED)


# ================================================
rootfinder_is = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.max_iter': 1000,
        # 'ipopt.hessian_approximation': 'limited-memory',
    },
)

x0_is = ntw.system.get_scaled_guess()
kn_is = ntw.system.get_scaled_constraints()
bnd_is = ntw.system.get_arguments_bounds()
solution = solve_root_problem(
    rootfinder_is,
    x0_is,
    kn_is,
    bnd_is,
    suppress_output=False,
    perturbate_guess=False,
    delta_pert=0.001,
    num_samples=1,
)

# ================================================
user = input('INPUT >>> Continue with losses? [y/n] ')
if user in ('y', 'Y'):
    # Write solution to dict for reading for next solution
    sol_dict_is = ntw.system.solution_to_dict(solution)

    # Remove the first computation loss
    ntw.system.remove_equation_type(LossApplier)

    rotor.remove_equation(INITIAL_LOSS.__class__, (0, 1))
    stator.remove_equation(INITIAL_LOSS.__class__, (0, 1))
    for eq, pos in EXTRA_EQUATIONS.items():
        rotor.add_equation(eq(), pos)
        stator.add_equation(eq(), pos)

    rotor.add_equation(LossMatcher(tip_gap=True), (0, 1))
    stator.add_equation(LossMatcher(tip_gap=False), (0, 1))
    ntw.build()

    x0_loss = ntw.system.get_scaled_guess(sol_dict_is)
    kn_loss = ntw.system.get_scaled_constraints()
    bnd_loss = ntw.system.get_arguments_bounds()
    err_on_fail = int(
        input('INPUT >>> Fail on rootfinding error? [0/1] '),
    )
    rootfinder_loss = ntw.system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': bool(err_on_fail),
        },
    )
    solution = solve_root_problem(
        rootfinder_loss,
        x0_loss,
        kn_loss,
        # bnd_loss,
        suppress_output=False,
    )
    # = = = = = = = = = =
# ________________________________________________
# ________________________________________________


final_sol_dict = ntw.system.write_solution_to_nodes(solution)
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
    for n0_idx, n1_idx in grouper(num_nodes, 2, incomplete='ignore'):
        inl_node = ntw.system.nodes[n0_idx]
        out_node = ntw.system.nodes[n1_idx]
        ax_chord = out_node.geo.get('chord_ax').to_base_units().magnitude[0]

        # Plot meridional profile
        is_stator = (n0_idx // 2) % 2 == 0
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

# A little staircase :)
nodes = {}
for i, node in enumerate(ntw.system.nodes):
    # For simpler access set n0, n1, n2, ...
    nodes[i] = node

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
#     print(f'delta_smass_mixing {n3.stc.smass - n2.stc.smass}')
#     print(f'Deviation {n3.kin.beta - n2.kin.beta}')
#
#     q = 0.5 * n2.stc.rhomass * n2.kin.W**2
#     w = n2.geo.pitch * np.cos(n2.geo.metal_angle)
#     cpb = (n2.oth.p_base - n2.stc.p) / q
#
#     zeta_inc = (
#         -(cpb * n2.geo.bld_thick) / w
#         + 2 * n2.oth.mom_thick / w
#         + ((n2.oth.disp_thick + n2.geo.bld_thick) / w) ** 2
#     )
#
#     zeta_actual = (n2.rlt.p - n3.rlt.p) / q
#
#     print(f'Incompressible vs actual zeta {zeta_inc}, {zeta_actual}')

answer = input('INPUT >>> Show plots? [y/n]')
if answer in ('Y', 'y'):
    plt.show(block=False)
    input('Press enter to close ')
plt.close('all')

print(f'Inlet mach is {inl_node.kin.mach}')

# globals().update(residual_debugger(MixingMomentumBalances(), [n6, n7]))
