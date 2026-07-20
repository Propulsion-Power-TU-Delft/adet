import logging

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow
from adet.components.blade_row import RowGeometry, VanelessDiffuser
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.base_equation import EquationConfig, LossApplier
from adet.equations.control_volumes import (
    ChokingArea,
    OptimalIncidence,
    ThroatConditions,
)
from adet.equations.definitions import (
    EffectiveBladeNumber,
    IsentropicProperties,
)
from adet.equations.geometrical import MinimalCamberLine
from adet.equations.nondimensional import (
    TotalTotalPressureRatio,
)
from adet.equations.utils import residual_debugger
from adet.fluid.settings import FluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.losses.basic import (
    IsentropicLink,
    ZeroDeviation,
)
from adet.losses.compressors import (
    AungierChoking,
    BackstromSlip,
    BladeLoadingCoppage,
    ClearanceJansen,
    DiskFricDailyNece,
    HydraulicQuantities,
    IncidenceGalvas,
    MixingJohnstonDean,
    RecirculationCoppage,
    SkinFrictionJansen,
)
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.interpolation import resample_linear
from adet.tools.loggers import setup_logger
from adet.tools.plotting import setup_mpl
from adet.variables import NodeVariables, ThermoVariables

logger = logging.Logger(__name__)
setup_logger(logger)

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)


plt.style.use('dark_background')
setup_mpl(
    {
        'font.family': 'serif',
        'font.size': 19,
        'text.usetex': True,
    }
)

NUM_SPAN = 1
ENABLE_LOSSES = True
RUN_MULTI = True
RUN_SPEEDLINES = True
SPDL_PTS = 150  # Number of speedline points
#
RUN_PLOTS = True  # plotting section
SHOW_PLOTS = True  # non-interactive testing
BOUNDS = {
    n0.stc.Pressure.Glob: (100.0, 1e6),
    n0.stc.Temperature.Glob: (100.0, 1e3),
    # n0.kin.MachThroat.Glob: (0.0, 1.1),
    n0.oth.ThrPressure.Glob: (10.0, 1e6),
    n0.oth.ThrTemperature.Glob: (100.0, 2e7),
    # n0.ndim.EtaTT.Glob: (0.5, 1.0),
    # n0.ndim.PRatioTT.Glob: (2.0, 7.0),
}

# +++ Fluid settings
fluid_model_real = FluidModel(
    DebugAbstractState('HEOS', 'Air'),  # This just counts the number of updates
)
fluid_model_ideal = FluidModel(
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
        input_pair=cp.PSmass_INPUTS,
        out_properties=(thrm.Enthalpy,),
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
        # dht_leakage1: n1.loss.Dht_leakage.Hint,
        dht_recirc1: n1.loss.Dht_recirculation.Hint,
        dht_disk1: n1.loss.Dht_disk.Hint,
        dht_choking1: n1.loss.Dht_choking.Hint,
        T_is1: n1.oth.Tis_tot.Hint,
        eta_tt1: n1.ndim.EtaTT.Hint,
        pt1: n1.tot.Pressure.Hint,
    ):
        # Channel losses
        dht_int = (
            0.0
            + dht_skin1
            + dht_incidence1
            + dht_mixing1
            + dht_loading1
            + dht_choking1
            + dht_clearance1
            + dht_recirc1
            + dht_disk1
            # + dht_lost1
        )
        dht_ext = 0.0  # + dht_disk1  # + dht_leakage1

        tot_hmass_is1 = self.eos(pt1, s0)

        delta_s = dht_int / T_is1

        work = ht1 - ht0

        # Residuals
        r1 = s1 - (s0 + delta_s)
        r2 = (work + dht_ext) * eta_tt1 - (tot_hmass_is1 - ht0)

        return r1, r2


# +++ Boundary conditions


EQS_ISENTROPIC = {
    ZeroDeviation(): 1,  # No slip
    IsentropicLink(): (0, 1),
}

EQS_WITH_LOSSES = {
    # *** SLIP
    BackstromSlip(): (0, 1),
    HydraulicQuantities(): (0, 1),
    # *** Auxiliary
    OptimalIncidence(): 0,
    # *** LOSS MODELS
    IncidenceGalvas(): (0, 1),
    BladeLoadingCoppage(): (0, 1),
    SkinFrictionJansen(): (0, 1),
    MixingJohnstonDean(): 1,
    ClearanceJansen(): (0, 1),
    DiskFricDailyNece(): (0, 1),
    RecirculationCoppage(): (0, 1),
    AungierChoking(): (0, 1),
    # ClearanceBrasz(): (0, 1),
    # RecirculationOh(): (0, 1),
    # LeakageAungier(): (0, 1),
    # LeakageLostWork(): (0, 1),
}

# *** Speedline definitions
R_HUB = 0.04064
R_TIP = 0.1076833
NUM_BLADES = 15
height = R_TIP - R_HUB

# *** NEW SPEEDLINES
SPEEDS = [16048, 18141, 19216, 20272]
MASS_CHOKES = [5.21816102, 5.35550058, 5.433622, 5.5155405]
MIN_MASS = [2.5, 3.18, 3.62, 3.87]

mass_limits = [(ms, 0.999 * mc) for ms, mc in zip(MIN_MASS, MASS_CHOKES)]
SPEED_LINES = dict(zip(SPEEDS, mass_limits))

# *** Metal angle definition
METAL_ANGLE = np.array(np.radians([-33, -44, -56]))
BLADE_THICKNESS = np.array([0.003048, 0.000762])

# *** Thickness distribution
angle_distr = resample_linear(METAL_ANGLE, NUM_SPAN + 1)
thick_distr = resample_linear(BLADE_THICKNESS, NUM_SPAN + 1)
rad_distr = np.linspace(R_HUB, R_TIP, NUM_SPAN + 1)


def convert_to_cell_centers(distr: NDArray):
    return np.array([(distr[i] + distr[i + 1]) / 2 for i, _ in enumerate(distr[:-1])])


angle_distr = convert_to_cell_centers(angle_distr)
thick_distr = convert_to_cell_centers(thick_distr)
rad_distr = convert_to_cell_centers(rad_distr)

throat_area = (
    height
    * NUM_BLADES
    / NUM_SPAN
    * np.sum(2 * np.pi * rad_distr / NUM_BLADES * np.cos(angle_distr) - thick_distr)
)
throat_area = 0.0196555

if NUM_SPAN == 1:
    angle_distr = np.array(np.radians([-44]))
    thick_distr = np.array([0.001905])

# +++ Inlet
inlet = Inlet(
    boundary_conditions={
        n0.tot.Pressure: 101352.9,
        n0.tot.Temperature: 288.16,
        n0.kin.FlowAngleAbs: Quantity(0.0, 'rad'),
        n0.oth.CumMassFlow: 4.3,
        # n0.kin.MachThroat: 1.0,
    }
)
# +++ Shafts
shaft = Shaft(
    omega=Quantity(SPEEDS[-1], 'rpm'),
    is_constrained=True,
)

casing = Shaft(
    omega=Quantity(0, 'rpm'),
    is_constrained=True,
)

# +++ Flow Components
impeller = BladeRow(
    name='rotor',
    shaft=shaft,
    bound_cond={
        # *** Node 0 ***
        # > Geometry
        # - Meridional
        n0.geo.Rhub: R_HUB,
        n0.geo.Rtip: R_TIP,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        n0.geo.ThroatArea: throat_area,
        # - Blade
        n0.geo.MetalAngle: angle_distr[NUM_SPAN // 2],
        n0.geo.MetalAngleHub: METAL_ANGLE[0],
        n0.geo.MetalAngleTip: METAL_ANGLE[-1],
        n0.geo.BldThick: np.average(thick_distr),
        # *** Node 1 ***
        # > Geometry
        n1.geo.MeridionalAngle: Quantity(90, 'deg'),
        n1.geo.Rmid: Quantity(0.2159, 'm'),
        n1.geo.Height: Quantity(0.01524, 'm'),
        n1.geo.MetalAngle: Quantity(-30, 'deg'),
        n1.geo.ThickByPitch: 0.02,
        n1.geo.ChordAx: Quantity(0.133879895, 'm'),
        n1.geo.NumBlades: NUM_BLADES,
        n1.geo.NumSplitters: NUM_BLADES,
        # > Loss coefficients contributors
        n0.oth.IncCoeff: 1.0,  # Incidence
        n1.geo.BackClearance: 0.001,
        n0.geo.TipClearance: Quantity(0.235, 'mm'),
        n1.geo.TipClearance: Quantity(0.304, 'mm'),
        n1.geo.AbsRoughness: Quantity(1.524, 'micron'),
        n1.oth.SlipFactCoeff: 5,
        # n1.oth.WorkLossCoeff: 0.3,
        n1.oth.BlLoadingCoeff: 0.9,
        n1.oth.MinWakeFrac: 0.4,
        n1.oth.MaxWakeFrac: 0.4,
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
        ThroatConditions(): 0,
        ChokingArea(): 0,
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
        # vaneless_diff,
    ],
)


# Overall efficiency
ntw_hecc.system.add_equation(TotalTotalPressureRatio(), (0, 1))
# ntw_hecc.system.add_equation(TotalTotalCompressionEfficiency(), (0, 3))
ntw_hecc.system.add_spanwise_constants(n0.kin.V_mer, n0.geo.HDistr)
impeller.set_spanwise_constant(n1.stc.Pressure)
vaneless_diff.set_spanwise_constant(n1.stc.Pressure)
ntw_hecc.build()


x0 = ntw_hecc.system.get_scaled_guess(fallback=0.5)
kn_hecc_is = ntw_hecc.system.get_scaled_constraints()
bnd_hecc_is = ntw_hecc.system.get_arguments_bounds(
    custom_bounds=BOUNDS,
    ignore_defaults=False,
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
# solution_hecc_is = solve_root_problem(rtfn_kin, solution_hecc_is, kn_hecc_is)
sol_is_dict = ntw_hecc.system.sol_to_dict(solution_hecc_is)
# globals().update(residual_debugger(OptimalIncidence(), [0], sol_is_dict))

if RUN_MULTI:
    print('*** SOLVING MULTISPAN ISENTROPIC***')
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    ntw_hecc.system.num_span = NUM_SPAN
    impeller.set_boundary_cond(n0.geo.MetalAngle, angle_distr)
    impeller.set_boundary_cond(n0.geo.BldThick, thick_distr)

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

    if ENABLE_LOSSES:
        print('*** SOLVING MULTISPAN WITH LOSSES***')
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # Remove isentropic and add losses
        for eq, pos in EQS_ISENTROPIC.items():
            impeller.remove_equation(eq.__class__, pos)
        for eq, pos in EQS_WITH_LOSSES.items():
            impeller.add_equation(eq, pos)

        # Remove fixed wake fraction -> Dynamic
        ntw_hecc.system.data.boun_cond.pop(n1.oth.WakeFrac)
        ntw_hecc.system.add_equation(LossPicker(), (0, 1))
        # ntw_hecc.system.add_equation(IsentropicLink(), (0, 1))
        ntw_hecc.build()

        rootfinder_hecc_loss = ntw_hecc.system.make_rootfinder(
            'ipopt',
            opts={'error_on_fail': False},
        )
        rtfn_kin = ntw_hecc.system.make_rootfinder('kinsol')
        rtfn_ip = ntw_hecc.system.make_rootfinder('ipopt')
        x0_loss = ntw_hecc.system.get_scaled_guess(sol_multi_dict, fallback=0.9)
        kn_loss = ntw_hecc.system.get_scaled_constraints()
        bnd_loss = ntw_hecc.system.get_arguments_bounds(custom_bounds=BOUNDS)
        solution_loss = solve_root_problem(
            rootfinder_hecc_loss,
            x0_loss,
            kn_loss,
            # bnd_loss,
            suppress_output=False,
        )
        sol_loss_dict = ntw_hecc.system.sol_to_dict(solution_loss)
        globals().update(residual_debugger(DiskFricDailyNece(), [0, 1], sol_loss_dict))

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
        #
        #         fontweight='bold',
        #     )
        #
        # ax.set_ylabel('Loss / (Work + dht_ext) [%]',  fontweight='bold')
        # # ax.set_title(
        # #     'Loss Breakdown at Design Point\n'
        # #     f'Shaft Work: {work:.0f} J/kg, External Losses: {dht_ext:.0f} J/kg',
        # #
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

        omega_idx = list(ntw_hecc.system.data.boun_cond.keys()).index(n1.kin.Omega)
        omega_scl = ntw_hecc.system.constraints_scaling[omega_idx]

        mf_idx = list(ntw_hecc.system.data.boun_cond.keys()).index(n0.oth.CumMassFlow)
        mf_scl = ntw_hecc.system.constraints_scaling[mf_idx]

        print('*** RUNNING SPEEDLINES ***')

        choke_idx = list(ntw_hecc.system.data.boun_cond.keys()).index(
            n1.oth.ChokeMassflow
        )
        choke_scl = ntw_hecc.system.constraints_scaling[choke_idx]

        # Find indices for loss components (active ones from LossPicker)
        loss_specs = [
            # Int
            n1.loss.Dht_skin,
            n1.loss.Dht_incidence,
            n1.loss.Dht_clearance,
            n1.loss.Dht_mixing,
            n1.loss.Dht_loading,
            # Ext
            n1.loss.Dht_disk,
            n1.loss.Dht_recirculation,
            n1.loss.Dht_leakage,
            n1.loss.Dht_choking,
        ]
        loss_indices = {}
        loss_scales = {}
        for loss_name in loss_specs:
            try:
                idx = ntw_hecc.system.data.free_args.index(loss_name)
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
        for idx, (omega, massflows) in enumerate(speed_lines.items()):
            pratios = []
            etas = []
            ttratio = []
            losses_dict = {name: [] for name in loss_indices.keys()}
            converged_count = 0
            kn[omega_idx] = np.array([omega / omega_scl])
            kn[choke_idx] = np.array([MASS_CHOKES[idx] / choke_scl])
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
                    sol = solve_root_problem(rtfn_kin, sol, kn, suppress_output=False)

                    sol_dict = ntw_hecc.system.sol_to_dict(sol)

                    pr = np.average(sol_dict[n1.ndim.PRatioTT])
                    eta = np.average(sol_dict[n1.ndim.EtaTT])
                    ttr = np.average(
                        (sol_dict[n1.tot.Temperature] - sol_dict[n0.tot.Temperature])
                        / sol_dict[n0.tot.Temperature]
                    )

                    if pr > 8 or eta > 1.0 or eta < 0.8:
                        raise RuntimeError  # Discard fake solutions

                    pratios.append(pr)
                    etas.append(eta)
                    ttratio.append(ttr)

                    # Store converged solution
                    all_converged_solutions.append((mf, pr, sol.copy()))

                    # Calculate specific work for normalization from nodes
                    work = np.mean(sol_dict[n1.tot.Enthalpy]) - np.mean(
                        sol_dict[n0.tot.Enthalpy]
                    )

                    # Extract loss values normalized by work
                    for loss_name, idx in loss_indices.items():
                        loss_val = sol[idx][0] * loss_scales[loss_name]
                        normalized_loss = (
                            loss_val / 1000
                        )  # / work if work > 0 else np.nan
                        losses_dict[loss_name].append(normalized_loss)

                    converged_count += 1
                except Exception:
                    # raise e from e
                    logger.warning(
                        f'Convergence failed at mf={mf:.3f} kg/s, '
                        f'omega={omega * 60 / (2 * np.pi):.0f} RPM'
                    )
                    pratios.append(np.nan)
                    etas.append(np.nan)
                    ttratio.append(np.nan)
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

            # Store only converged points in computed speedline data
            converged_mask = [not np.isnan(pr) for pr in pratios]
            converged_mf = [mf for mf, conv in zip(massflows, converged_mask) if conv]
            converged_pr = [
                float(pr) for pr, conv in zip(pratios, converged_mask) if conv
            ]
            converged_eta = [
                float(eta) for eta, conv in zip(etas, converged_mask) if conv
            ]
            converged_ttr = [
                float(ttr) for ttr, conv in zip(ttratio, converged_mask) if conv
            ]

            computed_speedline_data[f'{rpm:.0f}'] = {
                'massflows': converged_mf,
                'pratios': converged_pr,
                'etas': converged_eta,
                'ttratio': converged_ttr,
            }

            massflows_lbs = massflows * 2.2
            axs[0].plot(
                massflows,
                pratios,
                label=f'{rpm:.2f}',
                marker='o',
                markersize=5,
                linewidth=2,
            )
            axs[1].plot(
                massflows,
                etas,
                label=f'{rpm:.2f}',
                marker='s',
                markersize=5,
                linewidth=2,
            )

        # Pressure ratio vs mass flow
        axs[0].set_xlabel(
            'Mass flow [lbm/s]',
        )
        axs[0].set_ylabel(
            'Pressure ratio [−]',
        )
        axs[0].set_title('Compressor Map: Pressure Ratio', fontweight='bold')
        axs[0].legend(loc='best')
        axs[0].grid(True, alpha=0.3)

        axs[0].legend(loc='best')
        axs[0].grid(True, alpha=0.3)

        # Efficiency vs pressure ratio
        axs[1].set_xlabel(
            'Mass flow [lbm/s]',
        )
        axs[1].set_ylabel(
            'Total-to-total efficiency [−]',
        )
        axs[1].set_title(
            'Compressor Performance: $\\eta$ vs PR',
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
        design_rpm = SPEEDS[-1]
        design_rpm_str = f'{design_rpm:.0f}'

        # Check if design speedline converged
        if design_rpm_str in computed_speedline_data:
            design_data = computed_speedline_data[design_rpm_str]
            # Check if any points converged (at least one non-None eta)
            if any(eta is not None for eta in design_data['etas']):
                fig_21k, ax_21k = plt.subplots(figsize=(9, 7))

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
                    'recirculation': 'Recirculation',
                    'incidence': 'Incidence',
                    'chk': 'Choking',
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
                    loss_values_21k.append(loss_array_filtered)
                    label = loss_name.symbol.replace('dht_', '').replace('1', '')
                    label = NAMES[label]
                    loss_labels_21k.append(label)

                # Generate colors from colormap
                try:
                    colormap = plt.get_cmap('tab20')
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
                ax_21k.set_xlabel(
                    r'$\dot{m}$ [kg/s]',
                    fontsize=24,
                )
                ax_21k.set_ylabel(
                    r'$\Delta h_{t,\mathrm{loss}}$ [kJ/kg]',
                    fontsize=24,
                )
                # ax_21k.set_title(
                #     f'Loss Breakdown vs Mass Flow: {design_rpm:.0f} RPM (Des. Point)',
                #
                #     fontweight='bold',
                # )
                ax_21k.legend(loc='upper center', framealpha=0.5)
                ax_21k.grid(True, alpha=0.3, linestyle='--')
                ax_21k.tick_params(axis='both', labelsize=22)

                fig_21k.tight_layout()
                fig_21k.savefig(
                    'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\'
                    '\\presentations\\images\\loss_breakdown_hecc.svg'
                )
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
        from adet.tools.plotting import plot_velocity_triangles

        fig, axs = plt.subplots(2, 2, figsize=(8, 20))

        # Node pairs: (inlet, outlet) for each component
        node_pairs = [(n0, n1), (n1, n2)]
        for plot_idx, (inlet_n, outlet_n) in enumerate(node_pairs):
            for node_idx, n in enumerate([inlet_n, outlet_n]):
                ax = axs[plot_idx][node_idx]
                ax.set_aspect('equal')

                plot_velocity_triangles(
                    sol_multi_dict[n.kin.V_tan],
                    sol_multi_dict[n.kin.V_mer],
                    sol_multi_dict[n.kin.BladeSpeed],
                    sol_multi_dict[n.geo.RDistr],
                    ax,
                )

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_aspect('equal')

        # Plot meridional profile for impeller only
        inlet_n = n0
        outlet_n = n1

        r_in = float(sol_multi_dict[inlet_n.geo.Rmid][0])
        r_out = float(sol_multi_dict[outlet_n.geo.Rmid][0])
        height_in = float(sol_multi_dict[inlet_n.geo.Height][0])
        height_out = float(sol_multi_dict[outlet_n.geo.Height][0])
        mer_angle_in = float(sol_multi_dict[inlet_n.geo.MeridionalAngle][0])
        mer_angle_out = float(sol_multi_dict[outlet_n.geo.MeridionalAngle][0])
        axial_chord = float(sol_multi_dict[outlet_n.geo.ChordAx][0])

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

        ax.set_title('Meridional profile')
        ax.set_xlabel(r'$z$ / [m]')
        ax.set_ylabel(r'$r$ / [m]')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close('all')

        # spanwise = np.arange(len(solution_loss[n1.loss.Dht_loading]))
        # plt.stackplot(
        #     spanwise,
        #     n1.oth.delta_hmass_loading,
        #     n1.oth.delta_hmass_clearance,
        #     n1.oth.delta_hmass_skin,
        #     n1.oth.delta_hmass_mixing,
        #     n1.oth.delta_hmass_incidence,
        #     n1.oth.delta_hmass_recirc,
        #     n1.oth.delta_hmass_leakage,
        #     n1.oth.delta_hmass_disk,
        #     labels=[
        #         'loading',
        #         'clearance',
        #         'skin',
        #         'mixing',
        #         'incidence',
        #         'recirculation',
        #         'leakage',
        #         'disk',
        #     ],
        # )
        # plt.ylabel('Enthalpy loss [J / kg / K]')
        # plt.xlabel('Spanwise station []')
        # plt.legend(loc='upper left')
        # plt.grid()

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close('all')
