from adet.equations.variables import ThermoVariables, NodeVariables
import logging

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow
from adet.components.blade_row import VanelessDiffuser, plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.base_equation import LossApplier, EquationConfig
from adet.equations.control_volumes import FullIncidence
from adet.equations.definitions import (
    EffectiveBladeNumber,
    IsentropicProperties,
)
from adet.equations.geometrical import MinimalCamberLine
from adet.equations.nondimensional import (
    TotalTotalPressureRatio,
)
from adet.equations.utils import residual_debugger
from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.losses.basic import (
    PercentageEntropyLoss,
    ZeroDeviation,
)
from adet.losses.compressors import (
    BackstromSlip,
    BladeLoadingCoppage,
    ClearanceBrasz,
    DiskFricDailyNece,
    HydraulicQuantities,
    IncidenceGalvas,
    LeakageAungier,
    LeakageLostWork,
    MixingJohnstonDean,
    RecirculationOh,
    SkinFrictionJansen,
)
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.interpolation import resample_linear
from adet.tools.loggers import setup_logger

logger = logging.Logger(__name__)
setup_logger(logger)

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)


plt.rcParams.update(
    {
        'text.usetex': True,
        'font.family': 'serif',
    }
)

NUM_SPAN = 5
ENABLE_LOSSES = True
RUN_MULTI = True
RUN_SPEEDLINES = False
SPDL_PTS = 50  # Number of speedline points
RPM_DES = 21000  #
#
RUN_PLOTS = False  # plotting section
SHOW_PLOTS = True  # non-interactive testing

# +++ Shafts
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
thrm = ThermoVariables()
fluid_settings = FluidSettings(
    model=fluid_model_ideal,
    update_variables=(
        thrm.Pressure,
        thrm.Temperature,
    ),  # Thermodynamic iteration variables
    update_length=2,  # Single phase => Two update vars
)


class LossPicker(LossApplier):
    config = EquationConfig(
        manual_units=('Pa', 'J / kg'),
        input_pair=cp.PSmass_INPUTS,
        out_properties=(n0.stc.Enthalpy.Glob,),
    )

    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
        ht0: n0.tot.Enthalpy.Hint,
        ht1: n1.tot.Enthalpy.Hint,
        dht_skin1: n1.loss.Dht_skin.Hint,
        dht_loading1: n1.loss.Dht_loading.Hint,
        dht_clearance1: n1.loss.Dht_clearance.Hint,
        dht_mixing1: n1.loss.Dht_mixing.Hint,
        dht_incidence1: n1.loss.Dht_incidence.Hint,
        dht_leakage1: n1.loss.Dht_leakage.Hint,
        dht_recirc1: n1.loss.Dht_recirculation.Hint,
        dht_disk1: n1.loss.Dht_disk.Hint,
        dht_lost1: n1.loss.Dht_lost.Hint,
        T_is1: n1.oth.Tis_tot.Hint,
        eta_tt3: n3.ndim.EtaTT.Hint,
        p2: n2.tot.Pressure.Hint,
        p3: n3.tot.Pressure.Hint,
    ):
        # Channel losses
        dht_int = (
            0.0
            + dht_skin1
            + dht_incidence1
            + dht_clearance1
            + dht_mixing1
            + dht_leakage1
            + dht_loading1
            + dht_lost1
        )
        dht_ext = 0.0 + dht_leakage1 + dht_recirc1 + dht_disk1

        tot_hmass_is3 = self.eos(p3, s0)

        delta_s = dht_int / T_is1

        work = ht1 - ht0

        # Residuals
        r1 = s1 - (s0 + delta_s)
        r2 = (work + dht_ext) * eta_tt3 - (tot_hmass_is3 - ht0)

        return r1, r2


# +++ Boundary conditions
inlet = Inlet(
    boundary_conditions={
        n0.tot.Pressure: 101352.9,
        n0.tot.Temperature: 288.16,
        n0.kin.FlowAngleAbs: Quantity(0.0, 'rad'),
        n0.oth.CumMassFlow: 4.98,
    }
)


EQS_ISENTROPIC = {
    ZeroDeviation(): 1,  # No slip
    PercentageEntropyLoss(0.0): (0, 1),
}

EQS_WITH_LOSSES = {
    # *** SLIP
    BackstromSlip(): (0, 1),
    HydraulicQuantities(): (0, 1),
    # *** LOSS MODELS
    ClearanceBrasz(): (0, 1),
    SkinFrictionJansen(): (0, 1),
    BladeLoadingCoppage(): (0, 1),
    FullIncidence(): 0,
    IncidenceGalvas(): (0, 1),
    MixingJohnstonDean(): 1,
    RecirculationOh(): (0, 1),
    LeakageAungier(): (0, 1),
    LeakageLostWork(): (0, 1),
    DiskFricDailyNece(): (0, 1),
}

# *** Speedline definitions
SPEEDS = [21e3, 20e3, 19e3, 18e3]
MASS_CHOKES = [5.6, 5.3, 4.9, 4.6]
MIN_MASS = [4.07, 3.72, 3.5, 3.12]
mass_limits = [(ms, mc + 0.0) for ms, mc in zip(MIN_MASS, MASS_CHOKES)]
SPEED_LINES = dict(zip(SPEEDS, mass_limits))

# *** Metal angle definition
METAL_ANGLE = np.array([-30, -44, -53])
HEIGHT = 0.0640433
R_MID = 0.07416165
deltaH = HEIGHT / NUM_SPAN
r_min = R_MID - HEIGHT / 2
r_max = R_MID + HEIGHT / 2
rr = np.linspace(r_min + deltaH / 2, r_max - deltaH / 2, NUM_SPAN)
angle_values = -30 - 23 / (r_max - r_min) * (rr - r_min)

# *** Thickness distribution
BLADE_THICKNESS = np.array([0.003048, 0.000762])
angle_values = resample_linear(METAL_ANGLE, NUM_SPAN)
thick_distribution = resample_linear(BLADE_THICKNESS, NUM_SPAN)

if NUM_SPAN == 1:
    angle_values = np.array([-44])
    thick_distribution = np.array([0.002])

angle_distribution = Quantity(angle_values, 'deg')

# +++ Components
impeller = BladeRow(
    name='rotor',
    shaft=shaft,
    row_type='rotor',
    bound_cond={
        # *** Node 0
        # > Geometry
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        n0.geo.Rmid: R_MID,
        n0.geo.Height: HEIGHT,
        n0.geo.MetalAngle: Quantity(-44, 'deg'),
        n0.geo.MetalAngleHub: Quantity(-30, 'deg'),
        n0.geo.MetalAngleTip: Quantity(-53, 'deg'),
        n0.geo.BldThick: 0.002,
        n0.geo.TipClearance: Quantity(0.235, 'mm'),
        n0.oth.IncCoeff: 0.5,
        # *** Node 1
        # > Geometry
        n1.geo.MeridionalAngle: Quantity(90, 'deg'),
        n1.geo.Rmid: Quantity(0.2159, 'm'),
        n1.geo.Height: Quantity(0.01524, 'm'),
        n1.geo.MetalAngle: Quantity(-30, 'deg'),
        n1.geo.ThickByPitch: 0.02,
        n1.geo.BackClearance: 0.001,
        n1.geo.TipClearance: Quantity(0.304, 'mm'),
        n1.geo.ChordAx: Quantity(0.133879895, 'm'),
        n1.geo.NumBlades: 15,
        n1.geo.NumSplitters: 15,
        n1.geo.AbsRoughness: Quantity(1.524, 'micron'),
        # > Loss coefficients
        n1.oth.SlipFactCoeff: 2.5,
        n1.oth.WorkLossCoeff: 0.3,
        n1.oth.BlLoadingCoeff: 0.75,
        n1.oth.MinWakeFrac: 0.3,
        n1.oth.MaxWakeFrac: 0.65,
        n1.oth.WakeFrac: 0.3,
        n1.oth.ChokeMassflow: MASS_CHOKES[0],
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
    bound_cond={
        n1.geo.Rmid: Quantity(0.3055659, 'm'),
        n1.geo.HeightRatio: 1.0,
    },
    extra_equations={
        # PercTotalPressureLoss(0.03): (0, 1),  # 5% loss
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
ntw_hecc.system.add_spanwise_constants(n0.kin.V_mer, n0.geo.HDistr)
impeller.set_spanwise_constant(n1.stc.Pressure)
vaneless_diff.set_spanwise_constant(n1.stc.Pressure)

ntw_hecc.build()

x0 = ntw_hecc.system.get_scaled_guess(
    manual_values={n0.kin.FlowAngleRel.Glob: -0.5},
    fallback=0.5,
)
kn_hecc_is = ntw_hecc.system.get_scaled_constraints()
bnd_hecc_is = ntw_hecc.system.get_arguments_bounds(
    custom_bounds={
        n0.kin.V_mer.Glob: (10.0, 480.0),
        n0.kin.BladeSpeed.Glob: (0, 600),
        n0.kin.FlowAngleRel.Glob: (-1.48, 1.48),
        n0.kin.RelMach.Glob: (0.0, 1.04),
        n0.ndim.EtaTT.Glob: (0.5, 1.0),
        n0.ndim.PRatioTT.Glob: (2.0, 7.0),
    },
)

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
sol_is_dict = ntw_hecc.system.sol_to_dict(solution_hecc_is)

if RUN_MULTI:
    print('*** SOLVING MULTISPAN ISENTROPIC***')
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    ntw_hecc.system.num_span = NUM_SPAN
    impeller.set_boundary_cond(n0.geo.MetalAngle, angle_distribution)
    impeller.set_boundary_cond(n0.geo.BldThick, thick_distribution)

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

    sol_multi_dict = ntw_hecc.system.sol_to_dict(solution_hecc_multi)


if __name__ == '__main__':
    if RUN_MULTI and ENABLE_LOSSES:
        print('*** SOLVING MULTISPAN WITH LOSSES***')
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # Remove isentropic and add losses
        for eq, pos in EQS_ISENTROPIC.items():
            impeller.remove_equation(eq.__class__, pos)
        for eq, pos in EQS_WITH_LOSSES.items():
            impeller.add_equation(eq, pos)

        # Remove fixed wake fraction -> Dynamic
        ntw_hecc.system.data.boun_cond.pop(n1.oth.WakeFrac)
        ntw_hecc.system.add_equation(LossPicker(), (0, 1, 2, 3))
        ntw_hecc.build()

        rootfinder_hecc_loss = ntw_hecc.system.make_rootfinder(
            'ipopt',
            opts={'error_on_fail': False},
        )
        rtfn_kin = ntw_hecc.system.make_rootfinder('kinsol')
        rtfn_ip = ntw_hecc.system.make_rootfinder('ipopt')
        x0_loss = ntw_hecc.system.get_scaled_guess(sol_multi_dict, fallback=0.5)
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
        sol_loss_dict = ntw_hecc.system.sol_to_dict(solution_loss)
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

        # Loss breakdown bar plot at design point
        # if RUN_PLOTS:
        # fig, ax = plt.subplots(figsize=(10, 6))
        #
        # # Extract loss components from node (spanwise average)
        # loss_components = {
        #     'Skin Friction': np.mean(n1.oth.delta_hmass_skin),
        #     'Incidence': np.mean(n1.oth.delta_hmass_incidence),
        #     'Clearance': np.mean(n1.oth.delta_hmass_clearance),
        #     'Mixing': np.mean(n1.oth.delta_hmass_mixing),
        #     'Loading': np.mean(n1.oth.delta_hmass_loading),
        #     'Leakage': np.mean(n1.oth.delta_hmass_leakage),
        #     'Recirculation': np.mean(
        #         n1.oth.delta_hmass_recirc + n1.oth.delta_hmass_lost
        #     ),
        #     'Disk Friction': np.mean(n1.oth.delta_hmass_disk),
        # }
        #
        # # Calculate work
        # work = np.mean(n1.tot.hmass) - np.mean(n0.tot.hmass)
        #
        # # Calculate dht_ext
        # dht_ext = (
        #     np.mean(n1.oth.delta_hmass_leakage)
        #     + np.mean(n1.oth.delta_hmass_recirc)
        #     + np.mean(n1.oth.delta_hmass_disk)
        # )
        #
        # # Normalize by (work + dht_ext)
        # denominator = work + dht_ext
        # normalized_losses = {
        #     k: v / denominator * 100 for k, v in loss_components.items()
        # }
        #
        # # Create bar plot
        # names = list(normalized_losses.keys())
        # values = list(normalized_losses.values())
        # colormap = plt.get_cmap('viridis')
        #
        # colors = colormap(np.linspace(0, 1, len(names)))
        #
        # bars = ax.bar(
        #     names,
        #     values,
        #     color=colors,
        #     # edgecolor='black',
        #     # linewidth=1.5,
        # )
        #
        # # Add value labels on bars
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(
        #         bar.get_x() + bar.get_width() / 2.0,
        #         height,
        #         f'{height:.2f}%',
        #         ha='center',
        #         va='bottom',
        #         fontsize=10,
        #         fontweight='bold',
        #     )
        #
        # ax.set_ylabel('Loss / (Work + dht_ext) [%]', fontsize=12, fontweight='bold')
        # # ax.set_title(
        # #     'Loss Breakdown at Design Point\n'
        # #     f'Shaft Work: {work:.0f} J/kg, External Losses: {dht_ext:.0f} J/kg',
        # #     fontsize=12,
        # #     fontweight='bold',
        # # )
        # ax.grid(True, alpha=0.3, axis='y')
        # plt.xticks(rotation=45, ha='right')
        # fig.tight_layout()
        #
        # if SHOW_PLOTS:
        #     plt.show()
        # else:
        #     plt.close(fig)

    if RUN_SPEEDLINES:
        speed_lines = {
            rpm * 2 * np.pi / 60: np.linspace(k[0], k[1], SPDL_PTS)
            for rpm, k in SPEED_LINES.items()
        }

        omega_idx = ntw_hecc.system.constraints.index('kin_omega1')
        omega_scl = ntw_hecc.system.constraints_scaling[omega_idx]

        mf_idx = ntw_hecc.system.constraints.index('oth_cum_massflow0')
        mf_scl = ntw_hecc.system.constraints_scaling[mf_idx]

        print('*** RUNNING SPEEDLINES ***')

        choke_idx = ntw_hecc.system.constraints.index('oth_massflow_choke1')
        choke_scl = ntw_hecc.system.constraints_scaling[choke_idx]

        # Find indices for loss components (active ones from LossPicker)
        loss_names = [
            'oth_delta_hmass_skin1',
            'oth_delta_hmass_incidence1',
            'oth_delta_hmass_clearance1',
            'oth_delta_hmass_mixing1',
            'oth_delta_hmass_loading1',
            # Ext
            'oth_delta_hmass_disk1',
            'oth_delta_hmass_recirc1',
            'oth_delta_hmass_leakage1',
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
        computed_speedline_data = {}  # Store computed results for comparison
        all_converged_solutions = []  # Store all converged solutions

        sol = ntw_hecc.system.get_scaled_guess(sol_loss_dict)
        kn = ntw_hecc.system.get_scaled_constraints()
        for omega, massflows in speed_lines.items():
            pratios = []
            etas = []
            losses_dict = {name: [] for name in loss_indices.keys()}
            converged_count = 0
            kn[omega_idx] = np.array([omega / omega_scl])
            kn[choke_idx] = np.array([massflows[-1] / choke_scl])
            for mf in massflows:
                kn[mf_idx] = np.array([mf / mf_scl])

                # Find closest previous solution to use as initial guess
                # Prioritize mass flow proximity, use pr as tie-breaker
                if all_converged_solutions:
                    mf_range = max(massflows) - min(massflows) + 1e-6
                    pr_range = (
                        max(prev_pr for _, prev_pr, _ in all_converged_solutions)
                        - min(prev_pr for _, prev_pr, _ in all_converged_solutions)
                        + 1e-6
                    )
                    min_distance = float('inf')
                    closest_sol = sol
                    for prev_mf, _, prev_sol in all_converged_solutions:
                        # Normalize differences for balanced weighting
                        mf_diff_norm = (mf - prev_mf) / mf_range
                        pr_diff_norm = 0.0  # pr_diff not applicable yet
                        # Use Euclidean distance (currently just mf difference)
                        distance = abs(mf_diff_norm)
                        if distance < min_distance:
                            min_distance = distance
                            closest_sol = prev_sol
                    sol = closest_sol.copy()

                try:
                    sol = solve_root_problem(rtfn_kin, sol, kn, suppress_output=True)

                    sol_dict = ntw_hecc.system.solution_to_dict(sol)

                    pr = np.average(sol_dict['oth_pRatio_tt3'])
                    eta = np.average(sol_dict['oth_eta_tt3'])

                    if pr > 8 or eta > 1.0 or eta < 0.8:
                        raise RuntimeError

                    pratios.append(pr)
                    etas.append(eta)

                    # Store converged solution
                    all_converged_solutions.append((mf, pr, sol.copy()))

                    # Calculate specific work for normalization from nodes
                    n0 = ntw_hecc.system.nodes[0]
                    n1 = ntw_hecc.system.nodes[1]
                    work = np.mean(n1.tot.hmass) - np.mean(n0.tot.hmass)

                    # Extract loss values normalized by work
                    for loss_name, idx in loss_indices.items():
                        loss_val = sol[idx][0] * loss_scales[loss_name]
                        normalized_loss = loss_val / work if work > 0 else np.nan
                        losses_dict[loss_name].append(normalized_loss)

                    converged_count += 1
                except Exception:
                    logger.warning(
                        f'Convergence failed at mf={mf:.3f} kg/s, '
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

            # Store computed speedline data for comparison
            computed_speedline_data[f'{rpm:.0f}'] = {
                'massflows': massflows.tolist(),
                'pratios': [float(pr) if not np.isnan(pr) else None for pr in pratios],
                'etas': [float(eta) if not np.isnan(eta) else None for eta in etas],
            }

            massflows_lbs = massflows * 2.2
            axs[0].plot(
                massflows,
                pratios,
                label=f'{rpm / RPM_DES:.2f} N_des',
                marker='o',
                markersize=5,
                linewidth=2,
            )
            axs[1].plot(
                massflows,
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
        axs[1].set_title(
            'Compressor Performance: $\\eta$ vs PR',
            fontsize=12,
            fontweight='bold',
        )
        axs[1].legend(loc='best')
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        else:
            print('Speedline performance plots generated (not shown)')
            plt.close(fig)

        # Create dedicated plot for 21k speedline (if converged)
        design_rpm = RPM_DES
        design_rpm_str = f'{design_rpm:.0f}'

        # Check if design speedline converged
        if design_rpm_str in computed_speedline_data:
            design_data = computed_speedline_data[design_rpm_str]
            # Check if any points converged (at least one non-None eta)
            if any(eta is not None for eta in design_data['etas']):
                fig_21k, ax_21k = plt.subplots(figsize=(12, 7))

                closest_rpm = min(
                    loss_data_by_speed.keys(), key=lambda x: abs(x - design_rpm)
                )
                data_21k = loss_data_by_speed[closest_rpm]
                massflows_21k = data_21k['massflows']
                losses_21k = data_21k['losses']

                # Prepare data for stackplot
                loss_values_21k = []
                loss_labels_21k = []

                # Filter out non-converged points (where losses are NaN)
                first_loss_name = list(loss_indices.keys())[0]
                first_loss = np.array(losses_21k[first_loss_name])
                converged_mask = ~np.isnan(first_loss)
                massflows_21k_filtered = massflows_21k[converged_mask]

                NAMES = {
                    'recirc': 'Recirculation',
                    'incidence': 'Incidence',
                    'clearance': 'Tip clearance',
                    'mixing': 'Mixing',
                    'skin': 'Skin friction',
                    'loading': 'Blade loading',
                    'disk': 'Disk friction',
                    'leakage': 'Leakage',
                }
                for loss_name in loss_indices.keys():
                    loss_array = np.array(losses_21k[loss_name])
                    loss_array_filtered = loss_array[converged_mask]
                    loss_array_scaled = loss_array_filtered
                    loss_values_21k.append(100 * loss_array_filtered)
                    label = loss_name.replace('oth_delta_hmass_', '').replace('1', '')
                    label = NAMES[label]
                    loss_labels_21k.append(label)

                # Generate colors from colormap
                try:
                    colormap = plt.get_cmap('Dark2')
                except Exception:
                    colormap = plt.get_cmap('viridis')

                stackplot_colors = colormap(np.linspace(0, 1, len(loss_labels_21k)))

                # Create stackplot with enhanced styling
                ax_21k.stackplot(
                    massflows_21k_filtered,
                    *loss_values_21k,
                    labels=loss_labels_21k,
                    colors=stackplot_colors,
                    alpha=0.95,
                )
                ax_21k.set_xlabel(r'$\dot{m}$ [kg/s]', fontsize=28)
                ax_21k.set_ylabel(
                    r'$\Delta h_{t,\mathrm{loss}} / (h_{t,3} - h_{t,1})$ [\%]',
                    fontsize=28,
                )
                # ax_21k.set_title(
                #     f'Loss Breakdown vs Mass Flow: {design_rpm:.0f} RPM (Des. Point)',
                #     fontsize=14,
                #     fontweight='bold',
                # )
                ax_21k.legend(loc='upper center', fontsize=22, framealpha=0.95)
                ax_21k.grid(True, alpha=0.3, linestyle='--')
                ax_21k.tick_params(axis='both', labelsize=22)

                fig_21k.tight_layout()
                if SHOW_PLOTS:
                    plt.show()
                else:
                    print('Design speedline loss plot generated (not shown)')
                    plt.close(fig_21k)
            else:
                print(
                    f'Design speedline ({design_rpm:.0f} RPM) did not converge'
                    f', skipping plot'
                )

        # Save computed speedline data for comparison with experimental data
        import json
        import pathlib

        output_path = (
            pathlib.Path(__file__).parent.parent.parent.parent
            / 'data'
            / 'opencases'
            / 'nasa_hecc'
            / 'computed_speedline_data.json'
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(computed_speedline_data, f, indent=2)
        print(f'Computed speedline data saved to {output_path}')

    # ---------------- PLOT ---------------------
    if RUN_PLOTS and not RUN_SPEEDLINES:
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

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_aspect('equal')
        offset = 0.0
        impeller_outlet_x = None
        impeller_outlet_y = None

        for comp in plottable_components:
            if comp == vaneless_diff:
                continue
            inlet_node = comp.get_inlet_node(ntw_hecc)
            outlet_node = comp.get_outlet_node(ntw_hecc)
            if not inlet_node or not outlet_node:
                raise ValueError('missing nodes')

            lines = plot_from_nodes(
                inlet_node,
                outlet_node,
                False,
                offset,
                'k',
                ax,
            )

            MAIN_MARK_SIZE = 7
            STT_COLOR = '#b31529'
            # Only plot impeller outlet
            ax.plot(
                np.ones(NUM_SPAN) * offset,
                inlet_node.geo.rr,
                'o',
                color=STT_COLOR,
                markersize=MAIN_MARK_SIZE,
            )

            offset += outlet_node.geo.chord_ax[0]
            hh_points = (
                offset - outlet_node.geo.height / 2 + np.cumsum(outlet_node.geo.hh)
            )
            rr_points = hh_points - outlet_node.geo.hh / 2
            ax.plot(
                rr_points,
                outlet_node.geo.rr,
                'o',
                color=STT_COLOR,
                markersize=MAIN_MARK_SIZE,
            )
            ax.grid(alpha=0.4)
            # Store outlet location for inset
            impeller_outlet_x = rr_points
            impeller_outlet_y = outlet_node.geo.rr

        # Increase label and title font sizes
        ax.set_xlabel(
            r'$z$ [m]',
            fontsize=35,
            # fontweight='bold',
        )
        ax.set_ylabel(
            r'$r$ [m]',
            fontsize=35,
            # fontweight='bold',
        )
        # ax.set_title('Impeller Geometry', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=30)

        # Add inset axis for zoomed region
        axins = inset_axes(
            ax, width='40%', height='40%', loc='upper left', borderpad=1.5
        )
        axins.set_aspect('equal')

        # Plot zoomed region on inset
        inset_offset = 0.0
        for comp in plottable_components:
            if comp == vaneless_diff:
                continue
            inlet_node = comp.get_inlet_node(ntw_hecc)
            outlet_node = comp.get_outlet_node(ntw_hecc)

            # Plot geometry on inset
            plot_from_nodes(
                inlet_node,
                outlet_node,
                False,
                inset_offset,
                'k',
                axins,
            )

            inset_offset += outlet_node.geo.chord_ax[0]

            # Replot outlet points on inset
            if impeller_outlet_x is not None and impeller_outlet_y is not None:
                axins.plot(
                    impeller_outlet_x,
                    impeller_outlet_y,
                    'o',
                    color=STT_COLOR,
                    markersize=10,
                )

        # Set zoom region
        axins.set_xlim(0.125, 0.145)
        axins.set_ylim(0.21, 0.218)
        axins.tick_params(labelbottom=False, labelleft=False)
        # axins.set_xlabel('Axial [m]', fontsize=11, fontweight='bold')
        # axins.set_ylabel('Radial [m]', fontsize=11, fontweight='bold')
        # axins.set_title('Outlet Detail', fontsize=12, fontweight='bold')

        # Add grid to inset
        axins.grid(True, alpha=0.3)

        # Connect inset to main axis with lines
        mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5')

        fig.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close('all')

        globals().update(residual_debugger(FullIncidence(), [n0]))
        globals().update(residual_debugger(IncidenceGalvas(), [n0, n1]))
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
