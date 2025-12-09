from typing import Callable, cast

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


# TODO:
# Standardize the interface to these equations that utilize intermediate states
# e.g. you can just instance the equation with N intermediate states and recover
# them by just accessing an attribute, such as self.get_eos()
# - How can I specify the update pairs
# - Would be nice to reuse auto recognition of update variables

# == * == * == * ==
# Sick idea for workflow
# ----------------------
# - One or multiple equations (i-th), require a N_i of thermo qties along the component
# - Get max(N_i) => Choose two `int` (intermediate) update variables for vec updates
# - Add the intermediate variables as system free arguments
#    - ! Need to impose first and last element as equal to in and out node
#    - e.g. int_hmass#2->3|100 <== * Intermediate hmass (100 pts) between node 2 and 3
#    - Prefix int gives special treatment (for node recognition)
#    - Adjust the length of the initial guess based on the suffix
# - HS_eos(int_hmass#2->3|100, int_smass#2->3|100) -> intermediate rhomass, p, ...
# - Within each equation you can just call self.get_qty('rhomass', 0.25)
#     - 0.25 is the relative measure of the position for that specific component
#     - 0.25 over 4 points => int_p#2->3|100[round(0.25 * 4) = 1]
# == * == * == * ==


class DentonProfileLoss(LossModel):
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

    def __init__(
        self,
        fluid_model: ExternalFluidModel,
        scaling_factor: list[float] | None = None,
    ):
        """
        This requires intermediate state updates, meaning ad eos object has to be
        provided manually
        """
        self._fluid_model = fluid_model
        self._eos_callback = None
        super().__init__(scaling_factor)

    @staticmethod
    def _build_velocity_profile(
        xi_by_camb_len_A, xi_by_camb_len_B, k_prof, kin_W0, kin_W1
    ) -> tuple[cs.DM, cs.DM, cs.DM]:
        """
        Build the velocity proile. `xi` is the curvilinear coordinate
        along the camberline
        """
        # Positions
        xi_by_camb_len = cs.horzcat(
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
        W_distr_ss = cs.horzcat(1.0 * kin_W0, W_mid_ss, W_mid_ss, kin_W1)  # Suction
        W_distr_ps = cs.horzcat(0.0 * kin_W0, W_mid_ps, W_mid_ps, kin_W1)  # Pressure

        return xi_by_camb_len, W_distr_ss, W_distr_ps

    def _compute_thermo_distributions(self, rlt_hmass0, stc_smass0, W_distr):
        """
        Compute the pressure distribution from total enthalpy and entorpy
        at the inlet

        Note
        ----
        This approximates the flow as isentropic along the blade itself
        and is valid only for axial machines where total relative enthalpy
        is conserved
        """
        num_span = max(rlt_hmass0.shape)
        NUM_STREAM = 4

        if self._eos_callback is None:
            _eos_callback = CasadiEoS(
                f'Denton_HS_{id(self)}',
                self._fluid_model.eos_object,
                cp.HmassSmass_INPUTS,
                ['p', 'rhomass', 'T'],
                num_span,
            )
            # ! Manual typing annotation !
            self._eos_callback = cast(
                Callable[..., tuple[cs.DM, cs.DM, cs.DM]],
                _eos_callback,
            )

        stc_hmass_dst = [rlt_hmass0 - W_distr[:, i] ** 2 / 2 for i in range(NUM_STREAM)]

        # Extract p, T, and density distributions from abstract state
        p_list, rho_list, T_list = [], [], []
        for h in stc_hmass_dst:
            p, rho, T = self._eos_callback(h, stc_smass0)
            p_list.append(p)
            rho_list.append(rho)
            T_list.append(T)

        p_cat = cs.horzcat(*p_list)
        rho_cat = cs.horzcat(*rho_list)
        T_cat = cs.horzcat(*T_list)

        return p_cat, rho_cat, T_cat

    @staticmethod
    def _trapezoid(y, x):
        """Trapezoidal rule"""
        dx = x[:, 1:] - x[:, :-1]
        integrand = (y[:, :-1] + y[:, 1:]) * dx / 2
        return cs.sum2(integrand)

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
        p_ss, rho_ss, T_ss = self._compute_thermo_distributions(
            rlt_hmass0, stc_smass0, W_distr_ss
        )
        p_ps, rho_ps, T_ps = self._compute_thermo_distributions(
            rlt_hmass0, stc_smass0, W_distr_ps
        )

        # xi is the curvilinear coordinate along the chord
        xi_dimensional = xi_by_camb_len * geo_camb_len1

        # Trapezoidal integration (can't use np.trapezoidal for differentiability)
        # (trapezoidal rule is exact because everything is linear)
        # [Pa * m = N / m]
        pressure_integral = self._trapezoid(p_ps - p_ss, xi_dimensional)

        # Entropy generation from 2D viscous dissipation
        # [ kg / m**3 ] * [ m**3 / s**3 ] / [K] * [m]
        # = [ kg * m / s**3 / K ] = [ N / s / K ]
        entropy_integral_ps = self._trapezoid(
            oth_Cd_profile1 * rho_ps * W_distr_ps**3 / T_ps,
            xi_dimensional,
        )

        entropy_integral_ss = self._trapezoid(
            oth_Cd_profile1 * rho_ss * W_distr_ss**3 / T_ss,
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
    pressure_int = dl._trapezoid(p_ps - p_ss, xi_by_camb_len * camb_len)
    fig.show()
