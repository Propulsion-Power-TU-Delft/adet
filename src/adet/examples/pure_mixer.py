import logging

import jax
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.equations import EquationBase
from adet.equations.fundamental import Kinematics, TotalStaticMatching
from adet.equations.nondimensional import AbsoluteMachNumber, StaticPressRatio
from adet.equations.special import ThermoVarsAdder
from adet.equations.utils import residual_debugger
from adet.fluid.settings import FluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.registries import VariableBoundsRegistry, reset_registries
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables

logger = logging.getLogger(__name__)
setup_logger(logger, logging.INFO, logging.INFO)


def dev_osnaghi(pr, Ma, alpha_a, gamma):
    """
    pr = p1/pa, can be a vector
    """
    chi = gamma / (gamma - 1)

    numerator_linear = gamma / (1 - gamma) * pr * np.tan(alpha_a)

    discriminant = (1 - pr) * (2 * chi * Ma**2 - 1 - (gamma + 1) / (gamma - 1) * pr) + (
        chi * pr * np.tan(alpha_a)
    ) ** 2

    denominator = 1 + gamma * Ma**2 - pr

    tan_delta_alpha = np.where(
        pr > 1,  # shock: pressure rise
        (numerator_linear - np.sqrt(discriminant)) / denominator,  # expansion
        (numerator_linear + np.sqrt(discriminant)) / denominator,
    )

    return np.arctan(tan_delta_alpha)


def loss_denton(
    rhomass,
    W_in,
    beta_in,
    pitch,
    p_base,
    p_in,
    tot_p_in,
    bld_thick,
    mom_thick,
    disp_thick,
):
    q = 0.5 * rhomass * W_in**2
    w = pitch * np.cos(beta_in)
    cpb = (p_base - p_in) / q
    zeta = (
        -(cpb * bld_thick) / w + 2 * mom_thick / w + ((disp_thick + bld_thick) / w) ** 2
    )
    delta_tot_p = zeta * q

    return delta_tot_p / (tot_p_in - p_in)


def loss_osnaghi(pr, Ma, M1, gamma):
    """Y = pt2 - pt1 / (pt1 - p1)"""
    gmo_half = (gamma - 1) / 2
    gamma_ratio = gamma / (gamma - 1)

    num = (1 + gmo_half * Ma**2) ** gamma_ratio - pr * (
        1 + gmo_half * M1**2
    ) ** gamma_ratio
    den = (1 + gmo_half * Ma**2) ** gamma_ratio - 1

    return num / den


class BalanceEquations(EquationBase):
    def residual(
        self,
        kin_W0,
        kin_W1,
        stc_rhomass0,
        stc_rhomass1,
        kin_beta0,
        kin_beta1,
        geo_bld_thick0,
        oth_disp_thick0,
        oth_mom_thick0,
        oth_p_base0,
        rlt_hmass0,
        rlt_hmass1,
        stc_p0,
        stc_p1,
        geo_pitch0,
        kin_dev_angle1,
    ):
        opening = geo_pitch0 * np.cos(kin_beta0)

        mass_in = stc_rhomass0 * kin_W0 * (opening - geo_bld_thick0 - oth_disp_thick0)
        mass_out = stc_rhomass1 * kin_W1 * geo_pitch0 * np.cos(kin_beta1)

        r_mass = mass_in - mass_out

        # 1 *** X-Momentum
        mom_in_x = (
            mass_in * kin_W0
            + stc_p0 * (opening - geo_bld_thick0)
            + oth_p_base0 * geo_bld_thick0
            - stc_rhomass0 * oth_mom_thick0 * kin_W0**2
        )
        mom_out_x = mass_out * kin_W1 * np.cos(kin_dev_angle1) + stc_p1 * opening
        r_momx = mom_in_x - mom_out_x

        # 2 *** Y-Momentum
        # p_suct = stc_p1
        # area_y = safe_abs(geo_pitch0 * np.sin(kin_beta0))
        # mom_in_y = p_suct * area_y
        # mom_out_y = stc_p1 * area_y + mass_out * kin_W1 * np.sin(kin_dev_angle1)
        # r_momy = mom_in_y - mom_out_y

        r_energ = rlt_hmass0 - rlt_hmass1

        # 3 *** Supersonic vs. subsonic switch
        # r_choke = kin_W0 - stc_speed_sound0
        r_dev = kin_dev_angle1 - np.sign(kin_beta0) * (kin_beta0 - kin_beta1)

        return r_mass, r_momx, r_dev, r_energ


if __name__ == '__main__':
    reset_registries()
    sys = CasadiSystem(1)
    model = FluidModel(IdealGasState(1.4, 287, 1.8e-5))
    # model = FluidModel(DebugAbstractState('HEOS', 'air'))
    thrm = ThermoVariables()
    settings = FluidSettings(model, update_variables=(thrm.Pressure, thrm.Enthalpy))
    sys.fluid_settings = settings

    sys.add_equation(BalanceEquations(), (0, 1))

    sys.add_equation(Kinematics(), 0)
    sys.add_equation(Kinematics(), 1)
    sys.add_equation(ThermoVarsAdder(), 0)
    sys.add_equation(ThermoVarsAdder(), 1)
    sys.add_equation(TotalStaticMatching(), 0)
    sys.add_equation(TotalStaticMatching(), 1)
    sys.add_equation(AbsoluteMachNumber(), 0)
    sys.add_equation(AbsoluteMachNumber(), 1)
    sys.add_equation(StaticPressRatio(), (0, 1))

    n0 = NodeVariables(0)
    n1 = NodeVariables(1)

    sys.add_equalities(
        (n0.geo.RDistr, n1.geo.RDistr),
        (n0.kin.Omega, n1.kin.Omega),
    )

    bld_thick_val = 0.01
    sys.add_boundary_conditions(
        {
            n0.tot.Pressure: 3e5,
            n0.tot.Temperature: 400,
            n0.geo.RDistr: 0.1,
            n0.geo.Pitch: 1.0,
            n0.geo.BldThick: bld_thick_val,
            n0.oth.MomThick: bld_thick_val * 0.075,
            n0.oth.DispThick: bld_thick_val * 0.15,
            n0.oth.PBase: 1.3e5,
            n0.kin.Omega: 0.0,
            n0.kin.FlowAngleRel: Quantity(60, 'deg'),
            n0.kin.Mach: 1.0,
            n1.ndim.PRatio: 0.3,
        }
    )

    sys.build()
    VariableBoundsRegistry().from_dict(
        {
            'dev_angle': (0.0, 0.8),
            'V': (0.0, 1000),
            'W': (0.0, 1000),
        }
    )

    rtfn_ip = sys.make_rootfinder('ipopt', opts={'error_on_fail': True})
    x0 = sys.get_scaled_guess()
    kn = sys.get_scaled_constraints()
    bnd = sys.get_arguments_bounds()
    sol = solve_root_problem(rtfn_ip, x0, kn, bnd)

    sys.write_solution_to_nodes(sol)
    n0 = sys.nodes[0]
    n1 = sys.nodes[1]

    print(n0.kin)
    print(n1.kin)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    n0.kin.plot(n0.geo, 10, ax[0])
    n1.kin.plot(n1.geo, 10, ax[1])
    plt.show()

    # NOTE: Sweep functionality requires manual index mapping with new VarSpec API
    # To perform sweeps, build a mapping of VarSpec objects to their constraint indices
    # For now, basic solve is demonstrated above. Uncomment and adapt for sweep:
    # constraint_map = {spec: idx for idx, spec in enumerate(sys.data.boun_cond.keys())}
    # angle_idx = constraint_map[n0.kin.FlowAngleRel]
    # pRatio_idx = constraint_map[n1.ndim.PRatio]

    RUN_SWEEP = (
        False  # Simplified - set to True and implement index mapping above for sweep
    )
    N_PTS = 20
    ANGLES = [
        45,
        60,
        75,
    ]

    betas = [a * np.pi / 180 for a in ANGLES]
    pratios = np.linspace(0.3, 1.0, N_PTS)
    deviations = {a: [] for a in betas}
    loss_coeffs = {a: [] for a in betas}
    loss_coeffs_dnt = {a: [] for a in betas}
    mer_machs = {a: [] for a in betas}
    out_machs = {a: [] for a in betas}
    out_angles = {a: [] for a in betas}

    rtfn_kn = sys.make_rootfinder('kinsol', {'error_on_fail': True})
    res_func = sys.make_residual_function()

    if RUN_SWEEP:
        fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        for beta in betas:
            kn[angle_idx] = np.array([beta]) * sys.constraints_scaling[angle_idx]
            # First sweep with ipopt
            sol = solve_root_problem(rtfn_ip, sol, kn)
            for pr in pratios:
                kn[pRatio_idx] = np.array([pr]) * sys.constraints_scaling[pRatio_idx]
                sol = solve_root_problem(rtfn_kn, sol, kn)
                sys.write_solution_to_nodes(sol)
                n0 = sys.nodes[0]
                n1 = sys.nodes[1]
                out_mach = n1.kin.W / n1.stc.speed_sound

                deviations[beta].append(n0.kin.alpha - n1.kin.alpha)
                loss_coeffs[beta].append((n0.tot.p - n1.tot.p) / (n0.tot.p - n0.stc.p))
                mer_machs[beta].append(n1.kin.Wm / n1.stc.speed_sound)
                out_machs[beta].append(out_mach)
                out_angles[beta].append(n1.kin.beta)
                Y_dent = loss_denton(
                    n1.stc.rhomass,
                    n0.kin.W,
                    n0.kin.beta,
                    n0.geo.pitch,
                    n0.oth.p_base,
                    n0.stc.p,
                    n0.tot.p,
                    n0.geo.bld_thick,
                    n0.oth.mom_thick,
                    n0.oth.disp_thick,
                )
                loss_coeffs_dnt[beta].append(Y_dent)

            throat_mach = 1.0  # Value from n0.kin.Mach boundary condition

            dev_analytical = dev_osnaghi(
                pratios,
                throat_mach,
                beta,
                model.eos_object._gamma,
            )
            loss_osn = loss_osnaghi(
                pratios,
                throat_mach,
                np.concatenate(out_machs[beta]),
                model.eos_object._gamma,
            )

            label = f'alpha {beta * 180 / np.pi:.2f}'

            ax[0].set_title('Deviations')
            ax[0].plot(pratios, np.array(deviations[beta]) * 180 / np.pi, label=label)
            ax[0].plot(
                pratios,
                dev_analytical * 180 / np.pi,
                'o',
                label=label + ' Osnaghi',
            )

            ax[1].set_title('Loss Y [%]')
            ax[1].plot(pratios, np.array(loss_coeffs[beta]) * 100, label=label)
            # Analytical models
            ax[1].plot(
                pratios,
                np.array(loss_coeffs_dnt[beta]) * 100,
                'o',
                label=label + ' Denton',
            )
            ax[1].plot(
                pratios,
                loss_osn * 100,
                'o',
                label=label + ' Osnaghi',
            )

            ax[2].set_title('Oulet Machs')
            # ax[2].plot(pratios, mer_machs[alpha], label=label)
            ax[2].plot(pratios, out_machs[beta], label=label)

        jax.tree.map(
            lambda a: a.grid(alpha=0.8),
            ax.tolist(),
        )
        jax.tree.map(
            lambda a: a.legend(),
            ax.tolist(),
        )

        ax[0].set_ylim(0, 20)
        ax[1].set_ylim(0, 14)
        ax[2].set_ylim(0, 1.9)

        fig.tight_layout()
        fig.show()

    globals().update(residual_debugger(Kinematics(), [n0]))
