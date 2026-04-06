# === IMPORTS
from copy import deepcopy
import logging
from typing import Type

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, Inlet, Shaft
from adet.components import ComponentNetwork
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
    MeridionalGeometry,
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
from adet.equations.utils import get_midspan_idx, residual_debugger
from adet.fluid.settings import ExternalFluidModel
from adet.fluid.settings import FluidSettings
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.losses.leakage import DentonRectLeakage, DentonTrapLeakage
from adet.losses.mixing import DentonMixingLoss, SieverdingBasePressure
from adet.losses.profile import DentonRectProfile, DentonTrapProfile
from adet.losses.secondary import SecondaryBSM
from adet.registries import DefaultUnitsRegistry, GuessRegistry, VariableBoundsRegistry
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.iter import grouper
from adet.tools.loggers import setup_logger
from adet.tools.strings import change_idx, get_index

logger = logging.getLogger(__name__)

setup_logger(
    logger,
    logging.DEBUG,
    logging.INFO,
    banned_keywords=['STREAM', 'findfont', 'sBIT'],
)
plt.close('all')

# === SETTINGS
NUM_SPAN = 5
SCALED = True
PLOTS = True
PLOT_MARKERS = True
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
        'p_choke': 0.4 * abs_state.p(),
        'reactDegree_ts': 0.5,
        'p': abs_state.p(),
        'T': abs_state.T(),
        'hmass': abs_state.hmass(),
        'smass': abs_state.smass(),
        'rhomass': abs_state.rhomass(),
        'k_prof': 0.3,  # Profile loading
        'zweifelCoeff': 0.85,
        'num_blades': 30.0,
    }
)

# ================================================
# *** Variable bounds
_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
# _bounds_reg.ignore_defaults = True
_bounds_reg.from_dict(
    {
        'U': (-0.1, 200.0),  # Reduce the search area
        'Vm': (20.0, 150.0),  # Reduce the search area
        'num_blades': (1.0, 100.0),
        'delta_smass_.*': (0.0, 20.0),
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


class AddAxialLosses(LossApplier):
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
        oth_delta_smass_mixing1,
        oth_delta_smass_leakage1,
        oth_delta_smass_profile1,
        oth_delta_smass_secondary1,
    ):
        n_span = max(stc_smass0.shape)
        midspan = get_midspan_idx(stc_smass0)
        main_loss = (
            oth_delta_smass_mixing1
            + oth_delta_smass_profile1
            + oth_delta_smass_secondary1[midspan]
        )

        # Linear distribution of tip leakage
        leak_loss = np.linspace(0, 1, n_span) * oth_delta_smass_leakage1[-1]

        if self.tip_gap:
            return stc_smass1 - (stc_smass0 + main_loss + leak_loss)
        return stc_smass1 - (stc_smass0 + main_loss)


DUTY_COEFFS = {
    'oth_flowCoeff1': 0.6,
    'oth_ts_loadCoeff1': 3,
    'oth_volflowRatio1': 3.0,
    'oth_reactDegree_ts1': 0.3,
}


LOSS_MODELS: dict[
    Type[EquationBase],
    int | tuple[int, ...],
] = {
    # *** Blade row losses
    IsentropicProperties: (0, 1),
    SecondaryBSM: (0, 1),
    # Trapezoidals
    DentonTrapProfile: (0, 1),
    DentonTrapLeakage: (0, 1),
    # Rectangulars
    DentonMixingLoss: 1,
    ClearanceByHeight: 1,
    BladeBlockage: 1,
    BoundaryLayerRatios: 1,
    SieverdingBasePressure: (0, 1),
    ModifiedZweifel: (0, 1),
}

# ================================================
# *** Inlet conditions
inlet = Inlet(
    {
        'oth': {
            'cum_massflow': 10,
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'hubtipRatio': 0.9,
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
        ZeroDeviation(): 1,  # No deviation
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
stator.set_boundary_cond('geo_flare_angle1', Quantity(20, 'deg'))
rotor.set_boundary_cond('geo_flare_angle1', Quantity(20, 'deg'))
# *** Duty coefficients
rotor.bc_from_dict(DUTY_COEFFS)  # Duty coefficients at node 1

if __name__ == '__main__':
    # ================================================
    # Create network
    ntw = ComponentNetwork(
        fluid_settings,
        inlet,
        CasadiSystem(num_span=1),
        components=[stator, rotor],
    )

    rotor.set_spanwise_constant('geo_chord_ax1')
    stator.set_spanwise_constant('geo_chord_ax1', 'geo_hh0', 'kin_Vm0')
    rotor.copy_from_previous('geo_hh', 'geo_rr')
    rotor.remove_equation(MeridionalGeometry, 0)

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

    rtfn_kinsol = ntw.system.make_rootfinder('kinsol', opts={'max_iter': 10000})

    x0_is = ntw.system.get_scaled_guess()
    kn_is = ntw.system.get_scaled_constraints()
    bnd_is = ntw.system.get_arguments_bounds(
        {'kin_alpha0': (-0.7, 0.7)},
    )
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

    # ========================== LOSSES
    user = input('INPUT >>> Continue with losses? [y/n] ')
    if user in ('y', 'Y'):
        rotor.rm_boundary_cond('geo_num_blades1')
        rotor.set_boundary_cond('geo_zweifelCoeff1', 0.85)
        stator.rm_boundary_cond('geo_num_blades1')
        stator.set_boundary_cond('geo_zweifelCoeff1', 0.85)

        # --- Remove zero blockage
        rotor.remove_equation(ZeroBlockage, 1)
        stator.remove_equation(ZeroBlockage, 1)

        for eq, pos in LOSS_MODELS.items():
            stator.add_equation(eq(), pos)
            rotor.add_equation(eq(), pos)

        ntw.build()

        x0_loss = ntw.system.get_scaled_guess(sol_dict_is)
        kn_loss = ntw.system.get_scaled_constraints()
        bnd_loss = ntw.system.get_arguments_bounds()

        rootfinder_loss = ntw.system.make_rootfinder(
            'ipopt',
            opts={
                'error_on_fail': True,
                'ipopt.hessian_approximation': 'limited-memory',
            },
        )
        rtfn_kin = ntw.system.make_rootfinder('kinsol')

        try:
            solution = solve_root_problem(rootfinder_loss, x0_loss, kn_loss, bnd_loss)
            solution = solve_root_problem(rtfn_kin, solution, kn_loss)
        except RuntimeError:
            # Retry without bounds
            solution = solve_root_problem(rootfinder_loss, x0_loss, kn_loss)
            solution = solve_root_problem(rtfn_kin, solution, kn_loss)

        sol_dict_loss = ntw.system.write_solution_to_nodes(solution)

    user = input('INPUT >>> Continue with multispan? [y/n] ')
    if user in ('y', 'Y'):
        ntw.system.num_span = 3
        if NUM_SPAN > 1:
            rotor.add_equation(FreeVortexDistribution(), 1)
            stator.add_equation(FreeVortexDistribution(), 1)

        ntw.build()

        opts = {
            'error_on_fail': False,
            'ipopt.hessian_approximation': 'limited-memory',
        }

        rtfn_multi_ip = ntw.system.make_rootfinder('ipopt', opts=opts)
        rtfn_multi_kn = ntw.system.make_rootfinder('kinsol')

        x0 = ntw.system.get_scaled_guess(sol_dict_loss)
        kn = ntw.system.get_scaled_constraints()
        bnd = ntw.system.get_arguments_bounds()

        sol_multi = solve_root_problem(rtfn_multi_ip, x0, kn)
        sol_multi = solve_root_problem(
            rtfn_multi_kn, sol_multi, kn, suppress_output=True
        )
        sol_dict_multi = ntw.system.write_solution_to_nodes(sol_multi)

        # --- Remove the first computation loss
        rotor.remove_equation(INITIAL_LOSS.__class__, (0, 1))
        stator.remove_equation(INITIAL_LOSS.__class__, (0, 1))

        # --- Add loss applier function
        stator.add_equation(AddAxialLosses(tip_gap=False), (0, 1))
        rotor.add_equation(AddAxialLosses(tip_gap=True), (0, 1))

        # Make multi span
        ntw.system.num_span = NUM_SPAN
        ntw.build()

        x0 = ntw.system.get_scaled_guess(sol_dict_multi)
        kn = ntw.system.get_scaled_constraints()

        rtfn_final = ntw.system.make_rootfinder('ipopt', opts=opts)
        sol_multi = solve_root_problem(rtfn_final, x0, kn, suppress_output=True)
        sol_dict_multi = ntw.system.write_solution_to_nodes(sol_multi)
    else:
        NUM_SPAN = 1

    ntw.print_structure()

    user = input('INPUT >>> Compute speedline? [y/n] ')
    if user in ('y', 'Y'):
        # Remove and add variables
        TO_POP = {
            'geo': ['aspRatio', 'flare_angle', 'zweifelCoeff'],
        }
        TO_ADD = {
            'geo': [
                'height0',
                'height1',
                'metal_angle0',
                'metal_angle1',
                'chord_ax1',
                'num_blades1',
            ],
        }
        FINAL_DICT = sol_dict_loss

        # Remove duty coefficients
        [rotor.rm_boundary_cond(a) for a in DUTY_COEFFS]

        # Omega + midspan inlet
        shaft.omega = FINAL_DICT['kin_omega3'][0]
        shaft.is_constrained = True
        # Reassign to be re-read
        rotor.shaft = shaft

        inlet.boundary_conditions['geo']['rr_midspan'] = FINAL_DICT['geo_rr_midspan0']
        inlet.boundary_conditions['geo'].pop('hubtipRatio')
        inlet.boundary_conditions['kin'] = {'alpha': FINAL_DICT['kin_alpha0']}

        for state, args in TO_POP.items():
            for row in [stator, rotor]:
                # Allow incidence
                row.remove_equation(ZeroDeviation, 0)
                row.remove_equation(ModifiedZweifel, (0, 1))
                [row.outlet_bc[state].pop(k, None) for k in args]

        for state, args in TO_ADD.items():
            for row in [stator, rotor]:
                for a in args:
                    rel_arg = state + '_' + a
                    rel_idx = get_index(rel_arg)
                    abs_idx = row.network_maps[ntw][rel_idx]
                    abs_arg = change_idx(rel_arg, abs_idx)
                    print(rel_arg + '->' + abs_arg)
                    row._boundary_conditions[rel_idx][state][a[:-1]] = FINAL_DICT[
                        abs_arg
                    ][0]

        stator.remove_equation(INITIAL_LOSS.__class__, (0, 1))
        rotor.remove_equation(INITIAL_LOSS.__class__, (0, 1))
        stator.add_equation(AddAxialLosses(tip_gap=False), (0, 1))
        rotor.add_equation(AddAxialLosses(tip_gap=True), (0, 1))

        rotor.inlet_bc['geo'].pop('height')

        ntw = ComponentNetwork(
            fluid_settings, inlet, CasadiSystem(NUM_SPAN), [stator, rotor]
        )
        ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))

        ntw.build()

        rtfn = ntw.system.make_rootfinder(
            'ipopt',
            {
                'error_on_fail': True,
                'ipopt.max_wall_time': 3,
            },
        )

        design_mf = inlet.boundary_conditions['oth']['cum_massflow']
        DIR_MASS_SPACE = np.linspace(design_mf, design_mf * 1.1, 80)
        REV_MASS_SPACE = np.linspace(design_mf, design_mf / 1.1, 80)

        rtfn_kin = ntw.system.make_rootfinder('kinsol')
        massflows = []
        efficiencies = []
        eff_idx = ntw.system.free_args.index('oth_eta_tt3')

        mf_idx = ntw.system.constraints.index('oth_cum_massflow0')

        x0 = ntw.system.get_scaled_guess(FINAL_DICT)
        kn = ntw.system.get_scaled_constraints()
        kn[mf_idx] = np.array(
            [DIR_MASS_SPACE[0] / ntw.system.constraints_scaling[mf_idx]]
        )
        sol_off = solve_root_problem(rtfn, x0, kn)
        sol_off = solve_root_problem(rtfn_kin, x0, kn)
        ntw.system.write_solution_to_nodes(sol_off)

        for space in [DIR_MASS_SPACE, REV_MASS_SPACE]:
            sol1 = None
            for mf in space:
                mf_idx = ntw.system.constraints.index('oth_cum_massflow0')
                kn[mf_idx] = np.array([mf / ntw.system.constraints_scaling[mf_idx]])
                try:
                    if sol1 is None:
                        sol1 = solve_root_problem(rtfn_kin, sol_off, kn)
                    else:
                        sol1 = solve_root_problem(rtfn_kin, sol1, kn)
                    efficiencies.append(np.abs(sol1[eff_idx]))
                    massflows.append(mf)
                except RuntimeError:
                    break
    # ------------ PLOTS ------------
    if PLOTS:
        plt.style.use('dark_background')
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
            if PLOT_MARKERS:
                ax_merid.plot(NUM_SPAN * [offset], inl_node.geo.rr, 'o', color='r')
                ax_merid.plot(
                    NUM_SPAN * [offset] + ax_chord, out_node.geo.rr, 'o', color='r'
                )

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
                    'w',
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
        for mf_idx, (n0_idx, n1_idx) in enumerate(blade_rows):
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

            color = cmap(mf_idx / (len(ntw.components) - 0.8))  # pyright:ignore

            if mf_idx == 0:
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

    globals().update(residual_debugger(SecondaryBSM(), [n2, n3]))
