import logging
from pint import Quantity
import matplotlib.pyplot as plt
import numpy as np
import CoolProp as cp

from adet.equations.base_equation import EquationBase, LossApplier
from adet.equations.control_volumes import FullIncidence
from adet.equations.utils import residual_debugger
from adet.solution import solve_root_problem
from adet.assembly import CasadiSystem
from adet.components import BladeRow
from adet.components.blade_row import VanelessDiffuser, plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork

from adet.equations.definitions import (
    IsentropicProperties,
    EffectiveBladeNumber,
)
from adet.equations.geometrical import MinimalCamberLine
from adet.equations.nondimensional import (
    WorkCoefficient,
    TotalTotalPressureRatio,
)
from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState

from adet.losses.basic import (
    PercTotalPressureLoss,
    PercentageEntropyLoss,
    ZeroDeviation,
)

from adet.losses.compressors import (
    AmiranteDiffuserMomentum,
    BackstromSlip,
    BladeLoadingCoppage,
    ClearanceJansen,
    DiskFricDailyNece,
    IncidenceGalvas,
    LeakageAungier,
    MixingJohnstonDean,
    RecirculationOh,
    SkinFrictionJansen,
)
from adet.registries import GuessRegistry, VariableBoundsRegistry
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.interpolation import resample_linear
from adet.tools.loggers import setup_logger

logger = logging.Logger(__name__)
setup_logger(logger)

# This makes the missing guesses default to 1
_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
_bounds_reg.from_dict(
    {
        'Vm': (10.0, 480.0),
        'U': (0, 600),
        'beta': (-1.48, 1.48),
        'relmach': (0.0, 1.04),
        'eta_tt': (0.5, 1.0),
        'pRatio_tt': (2.0, 7.0),
        # 'delta_hmass_.*': (10.0, 2e4),
        # 'delta_hmass_recirc': (10.0, 1.4e4),  # This tends to diverge, bound it
        # 'T': (0.9 * 288.16, 3 * 288.16),
        # 'p': (0.8e5, 6e5),
    }
)

_greg = GuessRegistry()
_greg.reset()
_greg.from_dict(
    {
        'beta': -0.5,
        'gamma_pv': 1.4,
    }
)
_greg.set_fallback_value(0.5)  # Missing values defaults to 0.5

NUM_SPAN = 1
PLOTS = False  # Set to False to skip plotting section
ENABLE_LOSSES = True
RUN_MULTI = True
RUN_SPEEDLINES = True
SPDL_PTS = 20
RPM_DES = 21700
SHOW_PLOTS = True  # Set to False for non-interactive testing
# +++ Shaftskin_omega0 (node 0) is unknown,
shaft = Shaft(
    omega=Quantity(RPM_DES, 'rpm'),
    is_constrained=True,
)

casing = Shaft(
    omega=Quantity(0, 'rpm'),
    is_constrained=True,
)

# +++ Fluid settings
fluid_model_real = ExternalFluidModel(
    DebugAbstractState('HEOS', 'Air'),  # This just counts the number of updates
)
fluid_model_ideal = AnalyticalFluidModel(
    IdealGasState(1.4, 287, 2e-5),
)

fluid_settings = FluidSettings(
    model=fluid_model_ideal,
    update_variables=('p', 'T'),  # Thermodynamic iteration variables
    update_length=2,  # Single phase => Two update vars
)


class LossPicker(LossApplier):
    manual_units = ('J / kg', 'dimensionless')
    input_pair = cp.PSmass_INPUTS
    output_quantities = ('hmass',)

    def residual(
        self,
        tot_hmass0,
        tot_hmass1,
        oth_delta_hmass_skin1,
        oth_delta_hmass_loading1,
        oth_delta_hmass_clearance1,
        oth_delta_hmass_mixing1,
        oth_delta_hmass_incidence1,
        oth_delta_hmass_recirc1,
        oth_delta_hmass_leakage1,
        oth_delta_hmass_disk1,
        oth_eta_tt1,
        tot_p1,
        stc_smass0,
    ):
        tot_hmass_is1 = self.eos(tot_p1, stc_smass0)

        # Loss addition
        r1 = tot_hmass1 - (
            tot_hmass_is1
            + oth_delta_hmass_skin1
            + oth_delta_hmass_incidence1
            # ( WARN: SHOCK MISSING)
            + oth_delta_hmass_clearance1
            + oth_delta_hmass_mixing1
            + oth_delta_hmass_loading1
            # Internal
            # + oth_delta_hmass_disk1
            # + oth_delta_hmass_recirc1
            # + oth_delta_hmass_leakage1
        )

        # Efficiency computation

        eta_tt = (tot_hmass_is1 - tot_hmass0) / (tot_hmass1 - tot_hmass0)
        r2 = oth_eta_tt1 - eta_tt

        return r1, r2


# +++ Boundary conditions
inlet = Inlet(
    {
        'tot': {
            'p': 101352.9,
            'T': 288.16,
        },
        'kin': {
            'alpha': 0.0,
        },
        'oth': {
            'cum_massflow': 5.3,
        },
    },
)


EQS_ISENTROPIC = {
    ZeroDeviation(): 1,  # No slip
    PercentageEntropyLoss(0.0): (0, 1),
}

EQS_WITH_LOSSES = {
    # *** SLIP
    BackstromSlip(): (0, 1),
    # *** PICKERS
    LossPicker(): (0, 1),  # Apply losses
    # PercentageEntropyLoss(0.0): (0, 1),
    # *** LOSS MODELS
    ClearanceJansen(): (0, 1),
    SkinFrictionJansen(): (0, 1),
    BladeLoadingCoppage(): (0, 1),
    FullIncidence(): 0,
    IncidenceGalvas(): (0, 1),
    MixingJohnstonDean(): 1,
    RecirculationOh(): (0, 1),
    LeakageAungier(): (0, 1),
    DiskFricDailyNece(): (0, 1),
}


# - # - # - # - #
# Metal angle distribution
METAL_ANGLE = np.array([-30, -44, -53])
BLADE_THICKNESS = np.array([0.003048, 0.000762])
angle_values = resample_linear(METAL_ANGLE, NUM_SPAN)
thick_distribution = resample_linear(BLADE_THICKNESS, NUM_SPAN)

if NUM_SPAN == 1:
    angle_values = np.array([-44])
    thick_distribution = np.array([0.002])

angle_distribution = Quantity(angle_values, 'deg')


# - # - # - # - #

# +++ Components
impeller = BladeRow(
    name='rotor',
    shaft=shaft,
    row_type='rotor',
    in_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': Quantity(0.07416165, 'm'),
            'height': Quantity(0.0670433, 'm'),
            # *** Blades specs
            'metal_angle': Quantity(-44, 'deg'),
            'bld_thick': 0.002,
            'tip_clearance': Quantity(0.3048, 'mm'),
        },
        'oth': {
            'incCoeff': 0.5,
        },
    },
    out_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(90, 'deg'),
            'rr_midspan': Quantity(0.2159, 'm'),
            'height': Quantity(0.01524, 'm'),
            # *** Blades specs
            'metal_angle': Quantity(-30, 'deg'),
            'thick_by_pitch': 0.02,  # Thickness by pitch ratio
            'back_clearance': 0.001,  # Back face clearance
            'chord_ax': Quantity(0.133879895, 'm'),
            'num_blades': 15,
            'num_splitters': 15,
        },
        'oth': {
            # 'eta_tt': 0.821,  # Total total efficiency
            # For losses
            'slip_factCoeff': 3.2,
            'abs_roughness': Quantity(1.524, 'micron'),
            'bl_loadingCoeff': 0.75,
            # Mixing
            'minWake_frac': 0.3,
            'maxWake_frac': 0.5,
            'massflow_choke': 5.6,
            #
        },
    },
    extra_equations={
        # ZeroDeviation(): 1,
        MinimalCamberLine(): (0, 1),
        EffectiveBladeNumber(): 1,
        # *** Enthalpy based Losses
        IsentropicProperties(): (0, 1),
        # *** Blockage (optional)
        # Definitions
        # WorkCoefficient(): (0, 1),
        **EQS_ISENTROPIC,
    },
)

vaneless_diff = VanelessDiffuser(
    'diffuser',
    outlet_bc={
        'geo': {
            'rr_midspan': Quantity(0.3055659, 'm'),
            'heightRatio': 1.0,
        },
    },
    extra_equations={
        PercTotalPressureLoss(0.05): (0, 1),  # 5% loss
    },
)

ntw_hecc = ComponentNetwork(
    fluid_settings=fluid_settings,
    inlet=inlet,
    backend=CasadiSystem(num_span=1, scale_suffix='<|'),
    components=[
        impeller,
        vaneless_diff,
    ],
)


# Overall efficiency
ntw_hecc.system.add_equation(TotalTotalPressureRatio(), (0, 3))
ntw_hecc.system.add_spanwise_constants('kin_Vm0', 'geo_hh0')
impeller.set_spanwise_constant('stc_p1')
vaneless_diff.set_spanwise_constant('stc_p1')

ntw_hecc.build()

x0 = ntw_hecc.system.get_scaled_guess()
kn_hecc_is = ntw_hecc.system.get_scaled_constraints()
bnd_hecc_is = ntw_hecc.system.get_arguments_bounds()

# IPOPT is more robust, takes variable limits into account -> For 'bi-stable' solutions
# KINSOL is faster, sometimes converges on problems where ipopt struggles
rootfinder_hecc_is = ntw_hecc.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.max_wall_time': 10,
    },
)

rtfn_kin = ntw_hecc.system.make_rootfinder('kinsol')
print('*** SOLVING SINGLE SPAN ISENTROPIC***')
solution_hecc_is = solve_root_problem(
    rootfinder_hecc_is,
    x0,
    kn_hecc_is,
    bnd_hecc_is,
    suppress_output=False,
)
solution_hecc_is = solve_root_problem(rtfn_kin, solution_hecc_is, kn_hecc_is)
sol_is_dict = ntw_hecc.system.write_solution_to_nodes(solution_hecc_is)

if RUN_MULTI:
    print('*** SOLVING MULTISPAN ISENTROPIC***')
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    ntw_hecc.system.num_span = NUM_SPAN
    impeller.set_boundary_cond('geo_metal_angle0', angle_distribution)
    impeller.set_boundary_cond('geo_bld_thick0', thick_distribution)

    ntw_hecc.build()
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    rootfinder_hecc_multi = ntw_hecc.system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': False,
            'ipopt.max_iter': 1000,
            'ipopt.max_wall_time': 25,
        },
    )
    rtfn_kin = ntw_hecc.system.make_rootfinder('kinsol')
    x0_multi = ntw_hecc.system.get_scaled_guess(sol_is_dict)
    kn_hecc_multi = ntw_hecc.system.get_scaled_constraints()
    bnd_hecc_multi = ntw_hecc.system.get_arguments_bounds()
    solution_hecc_multi = solve_root_problem(
        rootfinder_hecc_multi,
        x0_multi,
        kn_hecc_multi,
        # bnd_hecc_multi,
        suppress_output=False,
    )
    solution_hecc_multi = solve_root_problem(
        rootfinder_hecc_multi,
        x0_multi,
        kn_hecc_multi,
        # bnd_hecc_multi,
        suppress_output=False,
    )

    sol_multi_dict = ntw_hecc.system.write_solution_to_nodes(solution_hecc_multi)


if __name__ == '__main__':
    if RUN_MULTI and ENABLE_LOSSES:
        print('*** SOLVING MULTISPAN WITH LOSSES***')
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # Remove isentropic and add losses
        for eq, pos in EQS_ISENTROPIC.items():
            impeller.remove_equation(eq.__class__, pos)
        for eq, pos in EQS_WITH_LOSSES.items():
            impeller.add_equation(eq, pos)

        ntw_hecc.build()

        rootfinder_hecc_loss = ntw_hecc.system.make_rootfinder(
            'ipopt',
            opts={'error_on_fail': False},
        )
        rtfn_kin = ntw_hecc.system.make_rootfinder('kinsol')
        x0_loss = ntw_hecc.system.get_scaled_guess(sol_multi_dict)
        kn_loss = ntw_hecc.system.get_scaled_constraints()
        bnd_loss = ntw_hecc.system.get_arguments_bounds()
        solution_loss = solve_root_problem(
            rootfinder_hecc_loss,
            x0_loss,
            kn_loss,
            bnd_loss,
            suppress_output=False,
        )
        solution_loss = solve_root_problem(rtfn_kin, solution_loss, kn_loss)
        sol_loss_dict = ntw_hecc.system.write_solution_to_nodes(solution_loss)
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

    if RUN_SPEEDLINES:
        SPEEDS = [21700, 20600, 19500, 18400]
        MASS_CHOKES = [5.6, 5.25, 4.8, 4.5]
        MIN_MASS = [m - 1.3 for m in MASS_CHOKES]

        mass_limits = [(ms, mc) for ms, mc in zip(MIN_MASS, MASS_CHOKES)]

        SPEED_LINES = dict(zip(SPEEDS, mass_limits))

        speed_lines = {
            rpm * 2 * np.pi / 60: np.linspace(k[0], k[1], SPDL_PTS)
            for rpm, k in SPEED_LINES.items()
        }
        ntw_hecc.build()
        kn = ntw_hecc.get_scaled_constraints()
        x0 = ntw_hecc.get_scaled_guess(sol_loss_dict)
        bnd = ntw_hecc.get_arguments_bounds()

        omega_idx = ntw_hecc.system.constraints.index('kin_omega1')
        omega_scl = ntw_hecc.system.constraints_scaling[omega_idx]

        mf_idx = ntw_hecc.system.constraints.index('oth_cum_massflow0')
        mf_scl = ntw_hecc.system.constraints_scaling[mf_idx]

        pr_idx = ntw_hecc.system.free_args.index('oth_pRatio_tt3')
        pr_scl = ntw_hecc.system.free_args_scaling[pr_idx]
        print('*** RUNNING SPEEDLINES ***')
        sol = None
        rtfn = ntw_hecc.system.make_rootfinder(
            'ipopt',
            opts={
                'error_on_fail': False,
                'ipopt.max_iter': 500,
            },
        )
        rtfn_kin = ntw_hecc.system.make_rootfinder('kinsol')

        # Find indices for additional parameters
        eta_idx = ntw_hecc.system.free_args.index('oth_eta_tt1')
        eta_scl = ntw_hecc.system.free_args_scaling[eta_idx]

        choke_idx = ntw_hecc.system.constraints.index('oth_massflow_choke1')
        choke_scl = ntw_hecc.system.constraints_scaling[choke_idx]

        # Find indices for loss components (active ones from LossPicker)
        loss_names = [
            'oth_delta_hmass_skin1',
            'oth_delta_hmass_incidence1',
            'oth_delta_hmass_clearance1',
            'oth_delta_hmass_mixing1',
            'oth_delta_hmass_loading1',
        ]
        loss_indices = {}
        loss_scales = {}
        for loss_name in loss_names:
            try:
                idx = ntw_hecc.system.free_args.index(loss_name)
                loss_indices[loss_name] = idx
                loss_scales[loss_name] = ntw_hecc.system.free_args_scaling[idx]
            except ValueError:
                # Loss variable might not be in free_args (could be pinned)
                pass

        fig, axs = plt.subplots(1, 2, figsize=(12, 7))
        loss_data_by_speed = {}  # Store losses for stackplot

        for omega, massflows in speed_lines.items():
            pratios = []
            etas = []
            losses_dict = {name: [] for name in loss_indices.keys()}
            converged_count = 0
            kn[omega_idx] = np.array([omega / omega_scl])
            kn[choke_idx] = np.array([massflows[-1] / choke_scl])
            for mf in massflows:
                kn[mf_idx] = np.array([mf / mf_scl])

                prev_sol = sol
                try:
                    # if sol is None:
                    #     sol = solve_root_problem(
                    #         rtfn, x0, kn, bnd, suppress_output=True
                    #     )
                    # else:
                    #     sol = solve_root_problem(
                    #         rtfn, sol, kn, bnd, suppress_output=True
                    #     )
                    sol = solve_root_problem(rtfn_kin, x0, kn)

                    pr = sol[pr_idx][0] * pr_scl
                    eta = sol[eta_idx][0] * eta_scl
                    pratios.append(pr)
                    etas.append(eta)

                    # Extract loss values
                    for loss_name, idx in loss_indices.items():
                        loss_val = sol[idx][0] * loss_scales[loss_name]
                        losses_dict[loss_name].append(loss_val)

                    converged_count += 1
                except Exception:
                    print(
                        f'  Warning: '
                        f'convergence failed at mf={mf:.3f} kg/s, '
                        f'omega={omega * 60 / (2 * np.pi):.0f} RPM'
                    )
                    pratios.append(np.nan)
                    etas.append(np.nan)
                    for loss_name in loss_indices.keys():
                        losses_dict[loss_name].append(np.nan)

            print(
                f'Speed {omega:.0f} RPM:'
                f' {converged_count}/{len(massflows)} points converged'
            )
            print(
                f'  Pressure ratios:'
                f' {[f"{pr:.4f}" if not np.isnan(pr) else "nan" for pr in pratios]}'
            )
            rpm = omega / 2 / np.pi * 60

            # Store loss data for this speedline
            loss_data_by_speed[rpm] = {
                'massflows': massflows,
                'losses': losses_dict,
            }

            axs[0].plot(
                massflows * 2.2,
                pratios,
                label=f'{rpm / RPM_DES:.2f} N_des',
                marker='o',
                markersize=5,
                linewidth=2,
            )
            axs[1].plot(
                massflows * 2.2,
                etas,
                label=f'{rpm / RPM_DES:.2f} N_des',
                marker='s',
                markersize=5,
                linewidth=2,
            )

        # Pressure ratio vs mass flow
        axs[0].set_xlabel('Mass flow [lbm/s]', fontsize=11)
        axs[0].set_ylabel('Pressure ratio [−]', fontsize=11)
        axs[0].set_title(
            'Compressor Map: Pressure Ratio', fontsize=12, fontweight='bold'
        )
        axs[0].legend(loc='best')
        axs[0].grid(True, alpha=0.3)

        axs[0].legend(loc='best')
        axs[0].grid(True, alpha=0.3)

        # Efficiency vs pressure ratio
        axs[1].set_xlabel('Mass flow [lbm/s]', fontsize=11)
        axs[1].set_ylabel('Total-to-total efficiency [−]', fontsize=11)
        # axs[1].set_ylim(0.79, 0.865)
        axs[1].set_title(
            'Compressor Performance: η vs PR', fontsize=12, fontweight='bold'
        )
        axs[1].legend(loc='best')
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        else:
            print('Speedline performance plots generated (not shown)')
            plt.close(fig)

        # Create stackplots for losses at each speedline
        num_speedlines = len(loss_data_by_speed)
        fig_loss, axs_loss = plt.subplots(
            num_speedlines, 1, figsize=(10, 4 * num_speedlines)
        )

        # Handle single speedline case (axs_loss won't be an array)
        if num_speedlines == 1:
            axs_loss = [axs_loss]

        for ax_idx, (rpm, data) in enumerate(sorted(loss_data_by_speed.items())):
            massflows = data['massflows']
            losses = data['losses']

            # Prepare data for stackplot: convert to arrays, handle NaNs
            loss_values = []
            loss_labels = []

            for loss_name in loss_indices.keys():
                loss_array = np.array(losses[loss_name])
                # Replace NaNs with 0 for stackplot
                loss_array = np.nan_to_num(loss_array, nan=0.0)
                loss_values.append(loss_array)
                # Clean up label name
                label = loss_name.replace('oth_delta_hmass_', '').replace('1', '')
                loss_labels.append(label)

            # Create stackplot
            ax = axs_loss[ax_idx]
            ax.stackplot(
                massflows * 2.2,
                *loss_values,
                labels=loss_labels,
                alpha=0.8,
            )
            ax.set_xlabel('Mass flow [lbm/s]', fontsize=11)
            ax.set_ylabel('Enthalpy loss [J/kg]', fontsize=11)
            ax.set_title(
                f'Loss Breakdown: {rpm / RPM_DES:.2f} N_des ({rpm:.0f} RPM)',
                fontsize=12,
                fontweight='bold',
            )
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)

        fig_loss.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        else:
            print('Loss stackplots generated (not shown)')
            plt.close(fig_loss)

    # ---------------- PLOT ---------------------
    if PLOTS:
        n0 = ntw_hecc.system.nodes[0]
        n1 = ntw_hecc.system.nodes[1]
        n2 = ntw_hecc.system.nodes[2]
        n3 = ntw_hecc.system.nodes[3]

        fig, axs = plt.subplots(2, 2, figsize=(8, 20))
        if len(ntw_hecc.components) > 2:
            plottable_components = ntw_hecc.components[1:]
        else:
            plottable_components = ntw_hecc.components

        for cmp_idx, comp in enumerate(plottable_components):
            inlet_node = comp.get_inlet_node(ntw_hecc)
            outlet_node = comp.get_outlet_node(ntw_hecc)

            if inlet_node is None or outlet_node is None:
                raise ValueError('Missing nodes')

            node_idx = 0
            for n in (inlet_node, outlet_node):
                ax = axs[cmp_idx][node_idx]

                ax.set_title(f'Node number {2 * cmp_idx + node_idx}')
                ax.set_aspect('equal')
                n.kin.plot(n.geo, 8, ax)

                node_idx += 1

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        offset = 0.0
        for comp in plottable_components:
            inlet_node = comp.get_inlet_node(ntw_hecc)
            outlet_node = comp.get_outlet_node(ntw_hecc)
            if not inlet_node or not outlet_node:
                raise ValueError('missing nodes')

            lines = plot_from_nodes(inlet_node, outlet_node, False, offset, 'k')

            offset += outlet_node.geo.chord_ax[0]

        fig.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close('all')

        globals().update(residual_debugger(AmiranteDiffuserMomentum(), [n2, n3]))
        spanwise = np.arange(len(n1.oth.delta_hmass_loading))
        plt.stackplot(
            spanwise,
            n1.oth.delta_hmass_loading,
            n1.oth.delta_hmass_clearance,
            n1.oth.delta_hmass_skin,
            n1.oth.delta_hmass_mixing,
            n1.oth.delta_hmass_incidence,
            n1.oth.delta_hmass_recirc,
            n1.oth.delta_hmass_leakage,
            n1.oth.delta_hmass_disk,
            labels=[
                'loading',
                'clearance',
                'skin',
                'mixing',
                'incidence',
                'recirculation',
                'leakage',
                'disk',
            ],
        )
        plt.ylabel('Enthalpy loss [J / kg / K]')
        plt.xlabel('Spanwise station []')
        plt.legend(loc='upper left')
        plt.grid()

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close('all')
