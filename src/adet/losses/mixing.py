"""Mixing losses downstream of turbomachinery blades"""

from pathlib import Path
from typing import Literal

import casadi as cs
import CoolProp as cp
import numpy as np

from adet.equations.base_equation import DeviationModel, EquationBase
from adet.equations.utils import minmax_bound, safe_abs, safe_if_else, safe_min
from adet.losses.base_loss import LossModel
from adet.tools.interpolation import make_casadi_interpolant

BLADE_PARAM = 2  # For Sieverding -> tmp, make this an input


class SieverdingBasePressure(EquationBase):
    manual_units = ('Pa',)

    def _get_base_pressure_interpolant(self, blade_type: Literal['conv', 'conv-div']):
        data_folder = Path(__file__).parents[3] / 'data'
        x = np.load(data_folder / f'sieverding_{blade_type}_xq.npy')
        y = np.load(data_folder / f'sieverding_{blade_type}_yq.npy')
        data = np.load(data_folder / f'sieverding_{blade_type}_zq.npy')
        return make_casadi_interpolant(x, y, data, 'base_pr', 'linear')

    def residual(self, tot_p0, stc_p1, oth_p_base1):
        # Detect array shapes
        num_span = max(tot_p0.shape)

        # `map` makes it a multi-dimensional function
        base_p_interpolant = self._get_base_pressure_interpolant('conv').map(num_span)

        # TODO: Double check this
        # Hardcoded blade parameter (=2)-> Check meaning
        # Add support for other blade parameters

        first_param = stc_p1 / tot_p0
        second_param = BLADE_PARAM * (stc_p1**0)  # it's just an array of 2s
        table_entry = cs.horzcat(first_param, second_param).T
        pb_by__ptin = base_p_interpolant(table_entry).T
        return oth_p_base1 - pb_by__ptin * tot_p0


class MixingMomentumBalances(EquationBase):
    """
    Balances of mass, momentum and energy for a mixing
    0 = Throat
    1 = Mixed out conditions
    """

    def residual(
        self,
        kin_W0,
        stc_rhomass0,
        geo_bld_thick0,
        geo_metal_angle0,
        oth_mom_thick0,
        oth_p_base0,
        stc_p0,
        stc_speed_sound0,
        stc_p1,
        kin_W_choke0,
        kin_W1,
        geo_pitch0,
        kin_beta0,
        kin_beta1,
        kin_dev_angle1,
        oth_ch_massflow0,
        stc_smass0,
        stc_smass1,
        geo_hh0,
        oth_delta_smass_mixing1,
    ):
        # Blockage enforced through effective area
        mf = oth_ch_massflow0 / geo_hh0

        opening = geo_pitch0 * np.cos(geo_metal_angle0)

        # 1 *** X-Momentum
        mom_in_x = (
            mf * kin_W0
            + stc_p0 * (opening - geo_bld_thick0)
            + oth_p_base0 * geo_bld_thick0
            - stc_rhomass0 * kin_W0**2 * oth_mom_thick0
        )
        mom_out_x = mf * kin_W1 * np.cos(kin_dev_angle1) + stc_p1 * opening
        r_momx = mom_in_x - mom_out_x

        # 2 *** Y-Momentum
        # p_suct = (stc_p0 + oth_p_base0) / 2
        p_suct = stc_p0
        area_y = safe_abs(geo_pitch0 * np.sin(kin_beta0))
        mom_in_y = p_suct * area_y
        mom_out_y = stc_p1 * area_y + mf * kin_W1 * np.sin(kin_dev_angle1)
        r_momy = (mom_in_y - mom_out_y) / mom_in_y

        # No deviation at subsonic outlet, choke otherwise
        # r_no_dev = kin_beta0 - kin_beta1
        r_choke = kin_W0 / stc_speed_sound0 - 1

        r_regime = safe_if_else(kin_W0 >= kin_W_choke0, r_choke, r_momy)

        # Delta smass for bounding
        r_delta = oth_delta_smass_mixing1 - (stc_smass1 - stc_smass0)

        # Positive metal angle => positive deviation reduces flow angle
        deviation = np.sign(geo_metal_angle0) * (kin_beta0 - kin_beta1)
        r_dev = kin_dev_angle1 - deviation

        return r_dev, r_momx, r_delta, r_regime


class SimplifiedMixingBalances(EquationBase):
    manual_units = ('dimensionless', 'Pa', 'J / kg / K')
    scaling_factor = (None, None, 0.01)

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
    def residual(
        self,
        kin_beta0,
        kin_relmach0,
        geo_metal_angle0,
    ):
        cos_beta = np.cos(geo_metal_angle0)  # > 0
        beta = safe_abs(geo_metal_angle0)  # > 0
        delta0_rad = beta - np.arccos(
            cos_beta * (1 + (1 - cos_beta) * (2 * beta / np.pi) ** 2)  # pyright:ignore
        )

        X = 2 * kin_relmach0 - 1
        delta_sub_rad = delta0_rad * (1 - 10 * X**3 + 15 * X**4 - 6 * X**5)

        deviation_rad = safe_if_else(
            kin_relmach0 <= 0.5,
            delta0_rad,
            delta_sub_rad,
        )

        deviation_rad = -np.sign(geo_metal_angle0) * deviation_rad

        return kin_beta0 - (geo_metal_angle0 + deviation_rad)


class AungierSimpleMixLoss(LossModel):
    input_pair = cp.HmassP_INPUTS
    output_quantities = ('smass',)
    manual_units = ('J / kg / K',)

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
    input_pair = cp.HmassP_INPUTS
    output_quantities = ('smass',)
    manual_units = ('J / kg / K',)

    def residual(
        self,
        # Thermo
        stc_p0,
        rlt_p0,
        stc_rhomass0,
        # Kinematics
        kin_W0,
        # Geometry
        geo_metal_angle0,
        geo_pitch0,
        geo_bld_thick0,
        # Boundary layer
        oth_p_base0,
        oth_mom_thick0,
        oth_disp_thick0,
        # Entropy production check
        rlt_hmass0,
        stc_smass0,
        stc_speed_sound0,
        kin_relmach0,
        oth_delta_smass_mixing0,
    ):
        # No deviation
        velocity = safe_if_else(kin_relmach0 >= 1, stc_speed_sound0, kin_W0)
        dyn_press = 0.5 * stc_rhomass0 * velocity**2  # Dynamic head
        zeta = incomp_mixing_zeta(
            dyn_press,
            stc_p0,
            geo_metal_angle0,
            geo_pitch0,
            geo_bld_thick0,
            oth_p_base0,
            oth_mom_thick0,
            oth_disp_thick0,
        )
        zeta = minmax_bound(zeta, 0.0, 1.0)

        rlt_p_loss = rlt_p0 - dyn_press * zeta
        smass_loss = self.eos(rlt_hmass0, rlt_p_loss)

        return oth_delta_smass_mixing0 - (smass_loss - stc_smass0)


class MinimalChoke(EquationBase):
    def residual(self, kin_W0, kin_W_choke0, kin_beta0, kin_beta1):
        no_dev = kin_beta1 - kin_beta0
        choke = kin_W0 - kin_W_choke0
        return safe_if_else(kin_W0 >= kin_W_choke0, choke, no_dev)
