"""Axial turbine loss models: Denton profile, leakage and mixing, and secondary-flow (BSM) losses."""

import casadi as cs
import CoolProp as cp
import numpy as np

from adet.equations.base_equation import (
    EquationConfig,
)
from adet.equations.utils import (
    minmax_bound,
    safe_if_else,
    safe_max,
    safe_min,
    trapezoid2,
)
from adet.losses.base_loss import LossModel
from adet.losses.turbines.shared import (
    baumgartner_mixing_zeta,
    incomp_mixing_zeta,
    rectangular_vel_profile,
    trapezoidal_vel_profile,
)
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


class DentonTrapLeakage(LossModel):
    config = EquationConfig(
        manual_units=('J / kg / K',),
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(thrm.Pressure, thrm.Density),
    )

    def residual(
        self,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        s0: n0.stc.Entropy.Hint,
        T_is1: n1.oth.Tis_stc.Hint,
        W0: n0.kin.W_mag.Hint,
        W1: n1.kin.W_mag.Hint,
        camb_len1: n1.geo.CamberLength.Hint,
        ch_mf1: n1.oth.ChanMassflow.Hint,
        xi_A1: n1.oth.XiCambLenA.Hint,
        xi_B1: n1.oth.XiCambLenB.Hint,
        k_prof1: n1.oth.ProfileLoading.Hint,
        disch_coeff1: n1.oth.DischCoeff.Hint,
        tip_clr1: n1.geo.TipClearance.Hint,
        ds_leak1: n1.loss.Ds_leakage.Hint,
    ):
        xi_by_camb_len, W_distr_ss, W_distr_ps = trapezoidal_vel_profile(
            xi_A1, xi_B1, k_prof1, W0, W1
        )

        p_ss, _ = self.eos(h_rlt0 - W_distr_ss**2 / 2, s0)
        p_ps, rho_ps = self.eos(h_rlt0 - W_distr_ps**2 / 2, s0)
        xi_dimensional = xi_by_camb_len * camb_len1

        delta_p = cs.fmax(p_ps - p_ss, 0.1)  # Avoid NaN in root
        # [kg / s / m]
        dm_by_dxi = disch_coeff1 * tip_clr1 * (2 * rho_ps * delta_p) ** 0.5

        # [ J / kg ] * [ kg / s / m ] * [m] = [ J / s ]
        leak_integral = trapezoid2(
            W_distr_ss**2 * (1 - W_distr_ps / W_distr_ss) * dm_by_dxi, xi_dimensional
        )

        # [ J / s ] * [ s / kg ] * [ 1 / K ] = [ J / kg / K ] OK
        return ds_leak1 - leak_integral / (ch_mf1 * T_is1)


class DentonRectLeakage(LossModel):
    config = EquationConfig(
        manual_units=('J / kg / K',),
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(thrm.Pressure, thrm.Density),
    )

    def residual(
        self,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        s0: n0.stc.Entropy.Hint,
        T_is1: n1.oth.Tis_stc.Hint,
        W0: n0.kin.W_mag.Hint,
        W1: n1.kin.W_mag.Hint,
        camb_len1: n1.geo.CamberLength.Hint,
        ch_mf1: n1.oth.ChanMassflow.Hint,
        k_prof1: n1.oth.ProfileLoading.Hint,
        disch_coeff1: n1.oth.DischCoeff.Hint,
        tip_clr1: n1.geo.TipClearance.Hint,
        ds_leak1: n1.loss.Ds_leakage.Hint,
    ):
        W_ss, W_ps = rectangular_vel_profile(W0, W1, k_prof1)

        p_ss, _ = self.eos(h_rlt0 - W_ss**2 / 2, s0)
        p_ps, rho_ps = self.eos(h_rlt0 - W_ps**2 / 2, s0)

        delta_p = cs.fmax(p_ps - p_ss, 0.1)  # Avoid NaN in root derivatives
        # [kg / s / m]
        dm_by_dxi = disch_coeff1 * tip_clr1 * (2 * rho_ps * delta_p) ** 0.5

        # [ J / kg ] * [ kg / s / m ] * [m] = [ J / s ]
        leak_integral = W_ss**2 * (1 - W_ps / W_ss) * dm_by_dxi * camb_len1

        # [ J / s ] * [ s / kg ] * [ 1 / K ] = [ J / kg / K ] OK
        return ds_leak1 - leak_integral / (ch_mf1 * T_is1)


class DentonMixingLoss(LossModel):
    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
        # manual_units=('J / kg / K',),
    )

    def residual(
        self,
        p0: n0.stc.Pressure.Hint,
        p_rlt0: n0.rlt.Pressure.Hint,
        rho0: n0.stc.Density.Hint,
        W0: n0.kin.W_mag.Hint,
        beta0: n0.geo.MetalAngle.Hint,
        pitch0: n0.geo.Pitch.Hint,
        t0: n0.geo.BldThick.Hint,
        p_base0: n0.oth.PBase.Hint,
        mom_thick0: n0.oth.MomThick.Hint,
        disp_thick0: n0.oth.DispThick.Hint,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        s0: n0.stc.Entropy.Hint,
        a0: n0.stc.SpeedSound.Hint,
        ds_mixing0: n0.loss.Ds_mixing.Hint,
    ):
        # No deviation
        velocity = safe_min(a0, W0)
        dyn_press = 0.5 * rho0 * velocity**2  # Dynamic head
        zeta = incomp_mixing_zeta(
            dyn_press,
            p0,
            beta0,
            pitch0,
            t0,
            p_base0,
            mom_thick0,
            disp_thick0,
        )
        zeta = minmax_bound(zeta, 0.0, 1.0)

        rlt_p1_loss = p_rlt0 - dyn_press * zeta
        smass1_loss = self.eos(h_rlt0, rlt_p1_loss)

        return ds_mixing0 - (smass1_loss - s0)


class DentonBaumgartnerMixingLoss(LossModel):
    """Denton's trailing-edge mixing loss using
    :func:`baumgartner_mixing_zeta`'s compressible base-pressure
    coefficient in place of :class:`DentonMixingLoss`'s incompressible
    Cpb."""

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def residual(
        self,
        p0: n0.stc.Pressure.Hint,
        p_rlt0: n0.rlt.Pressure.Hint,
        gamma0: n0.oth.GammaPV.Hint,
        rel_mach0: n0.kin.RelMach.Hint,
        beta0: n0.geo.MetalAngle.Hint,
        pitch0: n0.geo.Pitch.Hint,
        t0: n0.geo.BldThick.Hint,
        p_base0: n0.oth.PBase.Hint,
        mom_thick0: n0.oth.MomThick.Hint,
        disp_thick0: n0.oth.DispThick.Hint,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        s0: n0.stc.Entropy.Hint,
        ds_mixing0: n0.loss.Ds_mixing.Hint,
    ):
        # Floor only, to avoid division by zero at stagnation; no upper cap
        mach_floored = safe_max(rel_mach0, 0.05)
        dyn_press = 0.5 * gamma0 * p0 * mach_floored**2  # Compressible dynamic head

        zeta = baumgartner_mixing_zeta(
            p0,
            gamma0,
            mach_floored,
            beta0,
            pitch0,
            t0,
            p_base0,
            mom_thick0,
            disp_thick0,
        )

        rlt_p1_loss = p_rlt0 - dyn_press * zeta
        smass1_loss = self.eos(h_rlt0, rlt_p1_loss)

        return ds_mixing0 - (smass1_loss - s0)


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
class DentonRectProfInc(LossModel):
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
class DentonTrapProfile(LossModel):
    """
    Axial blade profile losses based on simplified pressure distribution.
    It should be able to be used for axial compressors and turbine blades,
    so far it has been tested for turbines (check the integral signs mainly)
    """

    config = EquationConfig(
        manual_units=('N', 'J / kg / K'),
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(thrm.Pressure, thrm.Density, thrm.Temperature),
    )

    def residual(
        self,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        s0: n0.stc.Entropy.Hint,
        W0: n0.kin.W_mag.Hint,
        W1: n1.kin.W_mag.Hint,
        Vt0: n0.kin.V_tan.Hint,
        Vt1: n1.kin.V_tan.Hint,
        ch_mf1: n1.oth.ChanMassflow.Hint,
        xi_A1: n1.oth.XiCambLenA.Hint,
        xi_B1: n1.oth.XiCambLenB.Hint,
        k_prof1: n1.oth.ProfileLoading.Hint,
        cd_prof1: n1.oth.CdProfile.Hint,
        h1: n1.geo.HDistr.Hint,
        camb_len1: n1.geo.CamberLength.Hint,
        stag1: n1.geo.Stagger.Hint,
        ds_prof1: n1.loss.Ds_profile.Hint,
    ):
        xi_by_camb_len, W_distr_ss, W_distr_ps = trapezoidal_vel_profile(
            xi_A1, xi_B1, k_prof1, W0, W1
        )

        # Static enthalpies
        stc_hmass_ss = h_rlt0 - W_distr_ss**2 / 2
        stc_hmass_ps = h_rlt0 - W_distr_ps**2 / 2

        # NOTE: Idea, make smass1 also an input and distribute
        # entropy (linearly?) between inlet and outlet
        p_ss, rho_ss, temp_ss = self.eos(stc_hmass_ss, s0)
        p_ps, rho_ps, temp_ps = self.eos(stc_hmass_ps, s0)

        # xi = coordinate along the chord
        xi_dimensional = xi_by_camb_len * camb_len1

        # Trapezoidal integration (can't use np.trapezoidal for differentiability)
        # (trapezoidal rule is exact because everything is linear)
        # [Pa * m = N / m]
        delta_p = p_ps - p_ss
        pressure_integral = trapezoid2(delta_p, xi_dimensional)

        # Entropy generation from 2D viscous dissipation
        # [ kg / m^3 ] * [ m^3 / s^3 ] / [K] * [m]
        # = [ kg * m / s^3 / K ] = [ N / s / K ]
        entropy_integral_ps = cd_prof1 * trapezoid2(
            rho_ps * W_distr_ps**3 / temp_ps,
            xi_dimensional,
        )
        entropy_integral_ss = cd_prof1 * trapezoid2(
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
        delta_Vt = cs.fabs(Vt1 - Vt0)
        r1 = ch_mf1 * delta_Vt - pressure_integral * h1 * np.cos(stag1)

        # 2. SPECIFIC entropy generation [J / kg / K]
        r2 = ds_prof1 - entropy_integral * h1 / ch_mf1

        return r1, r2


class DentonRectProfile(LossModel):
    """Full loss model"""

    config = EquationConfig(
        manual_units=('N', 'J / kg / K'),
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(thrm.Pressure, thrm.Density, thrm.Temperature),
    )

    def residual(
        self,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        s0: n0.stc.Entropy.Hint,
        W0: n0.kin.W_mag.Hint,
        W1: n1.kin.W_mag.Hint,
        Vt0: n0.kin.V_tan.Hint,
        Vt1: n1.kin.V_tan.Hint,
        ch_mf1: n1.oth.ChanMassflow.Hint,
        cd_prof1: n1.oth.CdProfile.Hint,
        k_prof1: n1.oth.ProfileLoading.Hint,
        h1: n1.geo.HDistr.Hint,
        camb_len1: n1.geo.CamberLength.Hint,
        stag1: n1.geo.Stagger.Hint,
        ds_prof1: n1.loss.Ds_profile.Hint,
    ):
        W_ss, W_ps = rectangular_vel_profile(W0, W1, k_prof1)

        # Static enthalpies
        stc_hmass_ss = h_rlt0 - W_ss**2 / 2
        stc_hmass_ps = h_rlt0 - W_ps**2 / 2

        p_ss, rho_ss, temp_ss = self.eos(stc_hmass_ss, s0)
        p_ps, rho_ps, temp_ps = self.eos(stc_hmass_ps, s0)

        # [Pa * m = N / m]
        delta_p = p_ps - p_ss
        pressure_integral = delta_p * camb_len1

        # Entropy generation from 2D viscous dissipation
        # [ kg / m^3 ] * [ m^3 / s^3 ] / [K] * [m]
        # = [ kg * m / s^3 / K ] = [ N / s / K ]
        entropy_integral_ps = cd_prof1 * rho_ps * W_ps**3 / temp_ps * camb_len1
        entropy_integral_ss = cd_prof1 * rho_ss * W_ss**3 / temp_ss * camb_len1

        entropy_integral = entropy_integral_ps + entropy_integral_ss

        # 1. Tangential momentum balance [N]
        delta_Vt = cs.fabs(Vt1 - Vt0)
        r1 = ch_mf1 * delta_Vt - pressure_integral * h1 * np.cos(stag1)

        # 2. SPECIFIC entropy generation [J / kg / K]
        r2 = ds_prof1 - entropy_integral * h1 / ch_mf1

        return r1, r2


class SecondaryBSM(LossModel):
    """
    Benner, M. W., Sjolander, S. A., and Moustapha, S. H., 2006, “An Empirical
    Prediction Method for Secondary Losses in Turbines—Part II: A New Second-
    ary Loss Correlation,” ASME J. Turbomach., 128(2), pp. 281–291.
    """

    config = EquationConfig(
        manual_units=('J / kg / K',),
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def residual(
        self,
        h1: n1.geo.Height.Hint,
        chord1: n1.geo.Chord.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        stag1: n1.geo.Stagger.Hint,
        disp_ew1: n1.oth.DispThickEW.Hint,
        p_rlt0: n0.rlt.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        s0: n0.stc.Entropy.Hint,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        ds_sec1: n1.loss.Ds_secondary.Hint,
    ):
        hgt_by_ch = h1 / chord1
        cos_ratio = np.cos(beta0) / np.cos(beta1)
        disp_by_H = disp_ew1 / h1

        Y_min2 = (0.038 + 0.41 * np.tanh(1.2 * disp_by_H)) / (
            np.cos(stag1) ** 0.5
            * cos_ratio
            * (hgt_by_ch * np.cos(beta1) / np.cos(stag1)) ** 0.55
        )

        Y_gtr2 = (0.052 + 0.56 * np.tanh(1.2 * disp_by_H)) / (
            np.cos(stag1) ** 0.5
            * cos_ratio
            * hgt_by_ch
            * (np.cos(beta1) / np.cos(stag1)) ** 0.55
        )

        loss_coeffY = safe_if_else(hgt_by_ch < 2.0, Y_min2, Y_gtr2)
        # Bound the loss coefficient to be >= 0
        loss_coeffY = safe_if_else(loss_coeffY > 0.0, loss_coeffY, 0.0)

        rlt_p_out = (p_rlt0 + loss_coeffY * p1) / (loss_coeffY + 1)

        return ds_sec1 - (self.eos(h_rlt0, rlt_p_out) - s0)
