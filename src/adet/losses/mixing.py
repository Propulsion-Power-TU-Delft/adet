"""Mixing losses downstream of turbomachinery blades"""

from pathlib import Path
from typing import Literal

import casadi as cs
import CoolProp as cp
import numpy as np

from adet.equations.base_equation import DeviationModel, EquationBase, EquationConfig
from adet.equations.utils import (
    minmax_bound,
    safe_abs,
    safe_if_else,
    safe_min,
)
from adet.losses.base_loss import LossModel
from adet.tools.interpolation import make_casadi_interpolant
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)
thrm = ThermoVariables()

BLADE_PARAM = 2  # TODO : For Sieverding -> make this an input


class SieverdingBasePressure(EquationBase):
    config = EquationConfig(
        manual_units=('Pa',),
    )

    def _get_base_pressure_interpolant(self, blade_type: Literal['conv', 'conv-div']):
        data_folder = Path(__file__).parents[3] / 'data'
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
        # Detect array shapes
        num_span = max(pt0.shape)

        # `map` makes it a multi-dimensional function
        base_p_interpolant = self._get_base_pressure_interpolant('conv').map(num_span)

        # TODO: Double check this
        # Hardcoded blade parameter (=2)-> Check meaning
        # Add support for other blade parameters

        first_param = p1 / pt0
        second_param = BLADE_PARAM * (p1**0)  # it's just an array of 2s
        table_entry = cs.horzcat(first_param, second_param).T
        pb_by__ptin = base_p_interpolant(table_entry).T
        return p_base1 - pb_by__ptin * pt0


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


class AungierDeviationModel(DeviationModel):
    """Only valid for subsonic deviation"""

    def residual(
        self,
        met_angle: n0.geo.MetalAngle.Hint,
        mach_out: n1.kin.RelMach.Hint,
        beta: n1.kin.FlowAngleRel.Hint,
    ):
        cos_beta = np.cos(met_angle)  # > 0
        beta_abs = safe_abs(met_angle)  # > 0
        delta0 = beta_abs - np.arccos(
            cos_beta * (1 + (1 - cos_beta) * (2 * beta_abs / np.pi) ** 2)
        )

        X = 2 * mach_out - 1
        delta_sub = delta0 * (1 - 10 * X**3 + 15 * X**4 - 6 * X**5)

        deviation_sub = safe_if_else(
            mach_out <= 0.5,
            delta0,
            delta_sub,
        )

        residual_sub = beta - (met_angle - deviation_sub)  # Deviation angle

        return residual_sub


class AungierSimpleMixLoss(LossModel):
    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def residual(
        self,
        stc_rhomass0,
        kin_W0,
        geo_pitch0,
        geo_metal_angle0,
        geo_bld_thick0,
        rlt_p0,
        rlt_hmass0,
        oth_disp_thick0,
        stc_smass0,
        oth_delta_smass_mixing0,
    ):
        opening = geo_pitch0 * np.cos(geo_metal_angle0)
        delta_pt = (
            0.5
            * stc_rhomass0
            * kin_W0**2
            * (opening / (opening - geo_bld_thick0 - oth_disp_thick0) - 1) ** 2
        )

        smass1 = self.eos(rlt_hmass0, rlt_p0 - delta_pt)

        return oth_delta_smass_mixing0 - (stc_smass0 - smass1)


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


class MinimalChoke(EquationBase):
    def residual(self, kin_W0, kin_W_choke0, kin_beta0, kin_beta1):
        no_dev = kin_beta1 - kin_beta0
        choke = kin_W0 - kin_W_choke0
        return safe_if_else(kin_W0 >= kin_W_choke0, choke, no_dev)
