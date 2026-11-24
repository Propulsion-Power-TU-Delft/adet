"""Turbine loss helpers shared across turbine types: profile velocity distributions, mixing balances and base pressure."""

from pathlib import Path
from typing import Literal

import casadi as cs
import numpy as np

from adet.equations.base_equation import EquationBase, EquationConfig
from adet.equations.utils import safe_abs, safe_if_else
from adet.tools.interpolation import make_casadi_interpolant
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)


BLADE_PARAM = 2  # TODO : For Sieverding -> make this an input


class SieverdingBasePressure(EquationBase):
    config = EquationConfig(
        manual_units=('Pa',),
    )

    def _get_base_pressure_interpolant(self, blade_type: Literal['conv', 'conv-div']):
        data_folder = Path(__file__).parent / 'data'
        x = np.load(data_folder / f'sieverding_{blade_type}_xq.npy')
        y = np.load(data_folder / f'sieverding_{blade_type}_yq.npy')
        data = np.load(data_folder / f'sieverding_{blade_type}_zq.npy')
        return make_casadi_interpolant(x, y, data, 'base_pr', 'linear')

    def residual(
        self,
        p1: n1.stc.Pressure.Hint,
        pt0: n0.rlt.Pressure.Hint,
        p_base1: n1.oth.PBase.Hint,
    ):
        return self._correlation_residual(p1, pt0, p_base1)

    def _correlation_residual(self, p1, pt0, p_base):
        # Detect array shapes
        num_span = max(pt0.shape)

        # `map` makes it a multi-dimensional function
        base_p_interpolant = self._get_base_pressure_interpolant('conv').map(num_span)

        # TODO: Double check this
        # Hardcoded blade parameter (=2)-> Check meaning
        # Add support for other blade parameters

        # Linearly extrapolates below the table's p1/pt0=0.25 lower bound
        # instead of clamping (validated against CFD by the model's author).
        first_param = p1 / pt0
        second_param = BLADE_PARAM * (p1**0)  # it's just an array of 2s
        table_entry = cs.horzcat(first_param, second_param).T
        pb_by__ptin = base_p_interpolant(table_entry).T
        return p_base - pb_by__ptin * pt0


class MixingMomentumBalances(EquationBase):
    """
    Balances of mass, momentum and energy for a mixing
    0 = Throat
    1 = Mixed out conditions
    """

    def residual(
        self,
        W0: n0.kin.W_mag.Hint,
        rho0: n0.stc.Density.Hint,
        bld_thick0: n0.geo.BldThick.Hint,
        metal_angle0: n0.geo.MetalAngle.Hint,
        mom_thick0: n0.oth.MomThick.Hint,
        dsp_thick0: n0.oth.DispThick.Hint,
        p_base0: n0.oth.PBase.Hint,
        p0: n0.stc.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        rlt_p0: n0.rlt.Pressure.Hint,
        rlt_p1: n1.rlt.Pressure.Hint,
        W1: n1.kin.W_mag.Hint,
        pitch0: n0.geo.Pitch.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        mf: n0.oth.StreamMassFlow.Hint,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
        mach0: n0.kin.RelMach.Hint,
        hh0: n0.geo.HDistr.Hint,
        dev_angle1: n1.kin.DevAngle.Hint,
        p_suct: n0.oth.PSuction.Hint,
        ds_mix1: n1.loss.Ds_mixing.Hint,
    ):
        # NOTE: Blockage is enforced through effective
        # area in mass conservation
        mf_by_h = mf / hh0

        outer_thr = pitch0 * np.cos(metal_angle0)

        # 1 *** X-Momentum
        mom_in_x = (
            mf_by_h * W0
            + p0 * (outer_thr - bld_thick0)
            + p_base0 * bld_thick0
            - rho0 * W0**2 * mom_thick0
        )
        mom_out_x = mf_by_h * W1 * np.cos(dev_angle1) + p1 * outer_thr
        r_momx = mom_in_x - mom_out_x

        # 2 *** Y-Momentum
        area_y = safe_abs(pitch0 * np.sin(beta0))
        mom_in_y = p_suct * area_y
        mom_out_y = p1 * area_y + mf_by_h * W1 * np.sin(dev_angle1)
        r_momy = (mom_in_y - mom_out_y) / mom_in_y

        q = 0.5 * rho0 * W0**2  # Dynamic head
        zeta = incomp_mixing_zeta(
            q,
            p0,
            metal_angle0,
            pitch0,
            bld_thick0,
            p_base0,
            mom_thick0,
            dsp_thick0,
        )
        # Actual loss application
        r_loss = 1 - (rlt_p0 - q * zeta) / rlt_p1

        # Positive metal angle => positive deviation reduces flow angle
        deviation = np.sign(metal_angle0) * (beta0 - beta1)
        r_dev = dev_angle1 - deviation

        r_choke = mach0 - 0.999
        r_switcher = safe_if_else(p1 / rlt_p0 >= 0.5297, r_momy, r_choke)  # noqa: F841

        # Entropy production for bounding
        r_ds = ds_mix1 - (s1 - s0)

        return (
            r_momx,
            r_momy,
            r_dev,
            r_ds,
            r_loss,
        )


class SimplifiedMixingBalances(EquationBase):
    config = EquationConfig(
        manual_units=('dimensionless', 'Pa', 'J / kg / K'),
        scaling_factor=(None, None, 0.01),
    )

    def residual(
        self,
        # Thermo
        stc_p0,
        rlt_p0,
        rlt_p1,
        stc_rhomass0,
        # Kinematics
        kin_W0,
        kin_beta0,
        kin_beta1,
        kin_relmach0,
        kin_relmach1,
        # Geometry
        geo_metal_angle0,
        geo_pitch0,
        geo_bld_thick0,
        # Boundary layer
        oth_p_base0,
        oth_mom_thick0,
        oth_disp_thick0,
        # Entropy production check
        stc_smass0,
        stc_smass1,
        oth_delta_smass_mixing1,
    ):
        # No deviation
        switch_supers = kin_relmach0 - 1.0  # Choking at throat
        switch_subson = kin_beta0 - kin_beta1  # Zero deviation
        r1 = cs.if_else(kin_relmach1 >= 0.9, switch_supers, switch_subson)

        q = 0.5 * stc_rhomass0 * kin_W0**2  # Dynamic head
        zeta = incomp_mixing_zeta(
            q,
            stc_p0,
            geo_metal_angle0,
            geo_pitch0,
            geo_bld_thick0,
            oth_p_base0,
            oth_mom_thick0,
            oth_disp_thick0,
        )

        r2 = rlt_p1 - (rlt_p0 - q * zeta)
        # Entropy production for bounding
        r3 = oth_delta_smass_mixing1 - (stc_smass1 - stc_smass0)

        return r1, r2, r3


def incomp_mixing_zeta(
    dyn_press,
    stc_p0,
    geo_metal_angle0,
    geo_pitch0,
    geo_bld_thick0,
    oth_p_base0,
    oth_mom_thick0,
    oth_disp_thick0,
):
    w = geo_pitch0 * np.cos(geo_metal_angle0)  # outlet throat
    cpb = (oth_p_base0 - stc_p0) / dyn_press

    return (
        -(cpb * geo_bld_thick0) / w
        + 2 * oth_mom_thick0 / w
        + ((oth_disp_thick0 + geo_bld_thick0) / w) ** 2
    )


def baumgartner_mixing_zeta(
    stc_p0,
    gamma_pv0,
    rel_mach0,
    geo_metal_angle0,
    geo_pitch0,
    geo_bld_thick0,
    oth_p_base0,
    oth_mom_thick0,
    oth_disp_thick0,
):
    """Denton mixing-loss coefficient with Baumgartner et al. (2020)'s
    compressible base-pressure-coefficient correction, replacing the raw
    incompressible ``Cpb = (p_base - p) / q`` used by
    :func:`incomp_mixing_zeta` with ``q = 0.5 * gamma_pv * p * Mach**2``.
    """
    thick_by_pitch0 = geo_bld_thick0 / geo_pitch0
    cpb = (
        2
        * (oth_p_base0 / stc_p0 - 1)
        * (2 / (gamma_pv0 * rel_mach0**2) * thick_by_pitch0)
    )

    w = geo_pitch0 * np.cos(geo_metal_angle0)  # outlet throat
    return (
        -(cpb * geo_bld_thick0) / w
        + 2 * oth_mom_thick0 / w
        + ((oth_disp_thick0 + geo_bld_thick0) / w) ** 2
    )


class MinimalChoke(EquationBase):
    def residual(self, kin_W0, kin_W_choke0, kin_beta0, kin_beta1):
        no_dev = kin_beta1 - kin_beta0
        choke = kin_W0 - kin_W_choke0
        return safe_if_else(kin_W0 >= kin_W_choke0, choke, no_dev)


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

    k_prof = safe_abs(k_prof)

    W_mid_ss = 2 * k_prof * kin_W1 + delta_W
    W_mid_ps = cs.fmax(k_prof * kin_W0 - delta_W, kin_W0 / MIN_CLIP)

    # Full velocity distribution
    W_distr_ss = cs.horzcat(1 * kin_W0, W_mid_ss, W_mid_ss, kin_W1)  # Suction
    # The 99 is needed otherwise the last state is equal, can cause NaN in some formulas
    W_distr_ps = cs.horzcat(0 * kin_W0, W_mid_ps, W_mid_ps, 0.99 * kin_W1)  # Pressure

    return xi_by_camb_len, W_distr_ss, W_distr_ps


def rectangular_vel_profile(kin_W0, kin_W1, oth_k_prof1):
    W_mean = (kin_W1 + kin_W0) / 2
    delta_W = safe_abs(kin_W1 - kin_W0)

    W_ss = W_mean + oth_k_prof1 * delta_W
    W_ps = W_mean - oth_k_prof1 * delta_W

    return W_ss, W_ps
