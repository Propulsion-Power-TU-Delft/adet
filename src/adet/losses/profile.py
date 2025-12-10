from typing import Callable, cast

from adet.equations.base_equation import MultiStateEquation
from adet.fluid.casadi_eos import CasadiEoS
import CoolProp as cp
import casadi as cs
import numpy as np

from adet.fluid.settings import ExternalFluidModel
from adet.losses.base_loss import LossModel
from adet.tools.coolprop_utils import DebugAbstractState


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
        kin_W0,
        kin_W1,
        oth_Cd_profile1,
        geo_camb_len1,
        geo_pitch1,
        tot_p0,
        tot_p1,
        stc_rhomass1,
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
#  |____________________________> Distance along blade
#               |          |
#            xi_by_camb_len1   xi_by_camb_len2


class DentonProfileLoss(MultiStateEquation):
    """
    Axial blade profile losses based on simplified pressure distribution.
    It should be able to be used for axial compressors and turbine blades,
    so far it has been tested for turbines (check the integral signs mainly)


    Warning
    -------
    - This function is ONLY compatible with CasADi
    - This assumes an external gas model
    """

    skip_unit_check = True
    manual_units = ('N', 'J / kg / K')
    input_properties = ('hmass', 'smass')
    output_properties = ('p', 'rhomass', 'T')

    def __init__(
        self,
        scaling_factor: list[float] | None = None,
    ):
        """
        This requires intermediate state updates, meaning ad eos object has to be
        provided manually
        """
        super().__init__(4, scaling_factor)

    @staticmethod
    def _build_velocity_profile(
        xi_by_camb_len_A, xi_by_camb_len_B, k_prof, kin_W0, kin_W1
    ) -> tuple[cs.DM, cs.DM, cs.DM]:
        """
        Build the velocity proile. `xi` is the curvilinear coordinate
        along the camberline
        """
        # Positions
        xi_by_camb_len = cs.vertcat(
            0.0 * xi_by_camb_len_A,
            xi_by_camb_len_A,
            xi_by_camb_len_B,
            xi_by_camb_len_A / xi_by_camb_len_A,
        )

        # Velocity values
        CLIP_RATIO = 5  # Clip pressure side to inlet W / ratio

        # NOTE: With the absolute value this should work for both diffusing
        # and accelerating channels
        delta_W = cs.fabs(kin_W1 - kin_W0)

        # NOTE: Nothing prevents k_prof from being negative, because
        # the values used in the residual formulation to compute the pressure
        # only use W**2.
        # The problem arises when you try to use the clip ratio, because you
        # are assuming the value of the sign by using the max function
        # => Solution = Take the absolute value of k_prof

        k_prof = cs.fabs(k_prof)  # Take the absolute value
        W_mid_ss = 2 * k_prof * kin_W1 + delta_W
        W_mid_ps = cs.fmax(k_prof * kin_W0 - delta_W, kin_W0 / CLIP_RATIO)

        # Full velocity distribution
        W_distr_ss = cs.vertcat(1.0 * kin_W0, W_mid_ss, W_mid_ss, kin_W1)  # Suction
        W_distr_ps = cs.vertcat(0.0 * kin_W0, W_mid_ps, W_mid_ps, kin_W1)  # Pressure

        return xi_by_camb_len, W_distr_ss, W_distr_ps

    def residual(
        self,
        # Thermo
        rlt_hmass0,
        stc_smass0,
        stc_smass1,
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
        geo_chord_ax1,
        geo_camb_len1,
        geo_stagger1,
        geo_num_blades1,
    ):
        xi_by_camb_len, W_distr_ss, W_distr_ps = self._build_velocity_profile(
            oth_xi_by_camb_len_A1, oth_xi_by_camb_len_B1, oth_k_prof1, kin_W0, kin_W1
        )

        # NOTE: Idea, make smass1 also an input and distribute
        # entropy (linearly?) between inlet and outlet
        p_ss, rho_ss, temp_ss = self.eos(rlt_hmass0 - W_distr_ss**2 / 2, stc_smass0)
        p_ps, rho_ps, temp_ps = self.eos(rlt_hmass0 - W_distr_ps**2 / 2, stc_smass0)

        # xi is the curvilinear coordinate along the chord
        xi_dimensional = xi_by_camb_len * geo_camb_len1

        # Trapezoidal integration (can't use np.trapezoidal for differentiability)
        # (trapezoidal rule is exact because everything is linear)
        # [Pa * m = N / m]
        pressure_integral = self.trapezoid(p_ps - p_ss, xi_dimensional)

        # Entropy generation from 2D viscous dissipation
        # [ kg / m**3 ] * [ m**3 / s**3 ] / [K] * [m]
        # = [ kg * m / s**3 / K ] = [ N / s / K ]
        entropy_integral_ps = self.trapezoid(
            oth_Cd_profile1 * rho_ps * W_distr_ps**3 / temp_ps,
            xi_dimensional,
        )

        entropy_integral_ss = self.trapezoid(
            oth_Cd_profile1 * rho_ss * W_distr_ss**3 / temp_ss,
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
        r2 = stc_smass1 - stc_smass0 - entropy_integral * geo_hh1 / oth_ch_massflow1

        return r1, r2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import casadi as cs
    import numpy as np

    # NOTE: You can mix and match cs.DM and
    # numpy array, but remember casadi is consistent
    # with shape manipulation and extraction, and every
    # array is AT LEAST 2D
    N_SPAN = 5
    eos = DebugAbstractState('HEOS', 'Air')
    model = ExternalFluidModel(eos)

    # Define example values
    W0 = np.linspace(100, 300, N_SPAN)
    W1 = np.linspace(175, 400, N_SPAN)

    Vt0 = np.linspace(40, 50, N_SPAN)
    Vt1 = np.linspace(40, 50, N_SPAN)

    xi_by_camb_len_A = cs.linspace(0.375, 0.375, N_SPAN)
    xi_by_camb_len_B = np.linspace(0.675, 0.675, N_SPAN)

    dummy_hh = cs.DM.ones(N_SPAN) * 0.05  # pyright:ignore
    dummy_ht = cs.DM.ones(N_SPAN) * 5e5  # pyright:ignore

    dummy_st = np.ones(N_SPAN) * 4000
    dummy_mass_flow = np.ones(N_SPAN) * 10

    Cd = 0.002 * np.ones(N_SPAN)
    chord_ax = 0.1 * np.ones(N_SPAN)
    camb_len = 0.15 * np.ones(N_SPAN)
    pitch = 0.2 * np.ones(N_SPAN)
    k_prof = 0.6 * np.ones(N_SPAN)

    # *** Tests ***
    dl = DentonProfileLoss(model)

    # Plots to check pressure and velocity distributions
    # => First station should clip to W_1 / 5
    xi_by_camb_len, W_ss, W_ps = dl._build_velocity_profile(
        xi_by_camb_len_A, xi_by_camb_len_B, k_prof, W0, W1
    )
    p_ss, rho_ss, T_ss = dl._compute_thermo_distributions(dummy_ht, dummy_st, W_ss)
    p_ps, rho_ps, T_ps = dl._compute_thermo_distributions(dummy_ht, dummy_st, W_ps)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    cmap = plt.get_cmap('viridis')
    for i in range(N_SPAN):
        color = cmap(i / (N_SPAN))
        ax[0, 0].plot(np.array(xi_by_camb_len)[i], np.array(W_ss)[i], color=color)
        ax[0, 0].plot(np.array(xi_by_camb_len)[i], np.array(W_ps)[i], color=color)
        ax[0, 0].set_title('Velocities')

        ax[0, 1].plot(np.array(xi_by_camb_len)[i], np.array(p_ss)[i], color=color)
        ax[0, 1].plot(np.array(xi_by_camb_len)[i], np.array(p_ps)[i], color=color)
        ax[0, 1].set_title('Pressures')

        ax[1, 0].plot(np.array(xi_by_camb_len)[i], np.array(rho_ss)[i], color=color)
        ax[1, 0].plot(np.array(xi_by_camb_len)[i], np.array(rho_ps)[i], color=color)
        ax[1, 0].set_title('Densities')

        ax[1, 1].plot(np.array(xi_by_camb_len)[i], np.array(T_ss)[i], color=color)
        ax[1, 1].plot(np.array(xi_by_camb_len)[i], np.array(T_ps)[i], color=color)
        ax[1, 1].set_title('Temperatures')

        ax[0, 0].grid(True)
        ax[0, 1].grid(True)
        ax[1, 0].grid(True)
        ax[1, 1].grid(True)

    # Let's check the integrals
    pressure_int = dl.trapezoid(p_ps - p_ss, xi_by_camb_len * camb_len)
    fig.show()
