import CoolProp as cp
import casadi as cs
import numpy as np

from adet.losses.base_loss import LossModel
from adet.equations.utils import minmax_bound, safe_abs, trapezoid2


# Greitzer model
# ==============
#                               Trailing edge
#  ^                  +---------------+--------  kin_W1
#  | Velocity         |               |
#  |                  | - - - - - - - |- W_mean
#  |                  |               |
#  |    kin_W0  ------+---------------+
#  |             Leading edge
#  |___________________________> Distance along blade
#
class RectVelocityIncompressible(LossModel):
    """
    References
    ----------
    - Section 5.4.3 of Greitzer's `Internal Flows`
    - Section 7.1 of Denton's 1993 hit paper `Loss Mechanisms in Turbomachines`
    """

    def residual(
        self,
        tot_p0,
        tot_p1,
        stc_rhomass1,
        #
        kin_W0,
        kin_W1,
        #
        geo_camb_len1,
        geo_pitch1,
        oth_Cd_profile1,
    ):
        delta_W = kin_W1 - kin_W0
        W_mean = (kin_W1 + kin_W0) / 2

        loss_coeff = (
            2
            * oth_Cd_profile1
            * (geo_camb_len1 / geo_pitch1)
            * (
                2 * (W_mean / kin_W0) ** 3
                + 6 * (W_mean / kin_W1) * (delta_W / kin_W1) ** 2
            )
        )

        return (tot_p0 - tot_p1) - loss_coeff * 0.5 * stc_rhomass1 * kin_W1**2


# Denton Model
# ============
#
#  ^              Suction
#  | Velocity     side
#  |            ___________ _ 2 k kin_W1 + delta_W
#  |           /           \
#  |          /             \ _ kin_W1
#  |       kin_W0           /
#  |            ___________/ _ k kin_W0 - delta_W
#  |           /  Pressure
#  |       0  /   side
#  |____________________________> Coordinate along camber
#               |          |
#     xi_by_camb_len_A    xi_by_camb_len_B


def trapezoidal_vel_profile(
    xi_by_camb_len_A, xi_by_camb_len_B, k_prof, kin_W0, kin_W1
) -> tuple[cs.DM, cs.DM, cs.DM]:
    """
    Build the velocity proile. `xi` is the curvilinear coordinate
    along the camberline

    Parameters
    ----------
    xi_by_camb_len_A
        position of the first point
    xi_by_camb_len_B
        position of the second point
    k_prof
        Velocity multiplier
    kin_W0
        Inlet velocity
    kin_W1
        Outlet velocity
    """
    # Positions
    xi_by_camb_len = cs.horzcat(
        0.0 * xi_by_camb_len_A,
        xi_by_camb_len_A,
        xi_by_camb_len_B,
        xi_by_camb_len_A / xi_by_camb_len_A,
    )

    # Velocity clippers
    MIN_CLIP = 5  # Clip pressure side to inlet W / <N>

    # NOTE: With the absolute value this should work for both diffusing
    # and accelerating channels
    delta_W = cs.fabs(kin_W1 - kin_W0)

    # NOTE: Nothing prevents k_prof from being negative, because
    # the values used in the residual formulation to compute the pressure
    # only use W**2.
    # The problem arises when you try to use the clip ratio, because you
    # are assuming the value of the sign by using the max function
    # => Solution = Take the absolute value of k_prof

    # Manual Bounding
    abs_k = safe_abs(k_prof)
    bnd_k = minmax_bound(abs_k, 0.01, 1.5)

    W_mid_ss = 2 * bnd_k * kin_W1 + delta_W
    W_mid_ps = cs.fmax(bnd_k * kin_W0 - delta_W, kin_W0 / MIN_CLIP)

    # Full velocity distribution
    W_distr_ss = cs.horzcat(1 * kin_W0, W_mid_ss, W_mid_ss, kin_W1)  # Suction
    # The 99 is needed otherwise the last state is equal, can cause NaN in some formulas
    W_distr_ps = cs.horzcat(0 * kin_W0, W_mid_ps, W_mid_ps, 0.99 * kin_W1)  # Pressure

    return xi_by_camb_len, W_distr_ss, W_distr_ps


class DentonProfileLoss(LossModel):
    """
    Axial blade profile losses based on simplified pressure distribution.
    It should be able to be used for axial compressors and turbine blades,
    so far it has been tested for turbines (check the integral signs mainly)
    """

    manual_units = ('N', 'J / kg / K')
    input_pair = cp.HmassSmass_INPUTS
    output_quantities = ('p', 'rhomass', 'T')

    def residual(
        self,
        # Thermo
        rlt_hmass0,
        stc_smass0,
        # Kine
        kin_W0,
        kin_W1,
        kin_Vt0,
        kin_Vt1,
        # Misc
        oth_ch_massflow1,
        oth_xi_by_camb_len_A1,
        oth_xi_by_camb_len_B1,
        oth_k_prof1,
        oth_Cd_profile1,
        # Geo
        geo_hh1,
        geo_camb_len1,
        geo_stagger1,
        oth_delta_smass_profile1,
    ):
        xi_by_camb_len, W_distr_ss, W_distr_ps = trapezoidal_vel_profile(
            oth_xi_by_camb_len_A1, oth_xi_by_camb_len_B1, oth_k_prof1, kin_W0, kin_W1
        )

        # Static enthalpies
        stc_hmass_ss = rlt_hmass0 - W_distr_ss**2 / 2
        stc_hmass_ps = rlt_hmass0 - W_distr_ps**2 / 2

        # NOTE: Idea, make smass1 also an input and distribute
        # entropy (linearly?) between inlet and outlet
        p_ss, rho_ss, temp_ss = self.eos(stc_hmass_ss, stc_smass0)
        p_ps, rho_ps, temp_ps = self.eos(stc_hmass_ps, stc_smass0)

        # xi is the curvilinear coordinate along the chord
        xi_dimensional = xi_by_camb_len * geo_camb_len1

        # Trapezoidal integration (can't use np.trapezoidal for differentiability)
        # (trapezoidal rule is exact because everything is linear)
        # [Pa * m = N / m]
        delta_p = p_ps - p_ss
        pressure_integral = trapezoid2(delta_p, xi_dimensional)

        # Entropy generation from 2D viscous dissipation
        # [ kg / m^3 ] * [ m^3 / s^3 ] / [K] * [m]
        # = [ kg * m / s^3 / K ] = [ N / s / K ]
        entropy_integral_ps = oth_Cd_profile1 * trapezoid2(
            rho_ps * W_distr_ps**3 / temp_ps,
            xi_dimensional,
        )
        entropy_integral_ss = oth_Cd_profile1 * trapezoid2(
            rho_ss * W_distr_ss**3 / temp_ss,
            xi_dimensional,
        )

        entropy_integral = entropy_integral_ps + entropy_integral_ss
        # NOTE: Integral = Entropy production (NOT SPECIFIC)
        # per unit length (in height direction) per unit time
        # To convert to specific (see residual definition below):
        #    |> multiply by height sector
        #    |> divide my massflow of each channel

        # 1. Tangential momentum balance [N]
        delta_Vt = cs.fabs(kin_Vt1 - kin_Vt0)
        r1 = oth_ch_massflow1 * delta_Vt - pressure_integral * geo_hh1 * np.cos(
            geo_stagger1
        )

        # 2. SPECIFIC entropy generation [J / kg / K]
        r2 = oth_delta_smass_profile1 - entropy_integral * geo_hh1 / oth_ch_massflow1

        return r1, r2
