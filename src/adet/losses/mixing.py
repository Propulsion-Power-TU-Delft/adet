"""Mixing losses downstream of turbomachinery blades"""

from pathlib import Path
from typing import Literal

import casadi as cs
import numpy as np

from adet.equations.base_equation import EquationBase
from adet.equations.utils import safe_abs
from adet.tools.interpolation import make_casadi_interpolant


class SieverdingBasePressure(EquationBase):
    manual_units = ('Pa',)

    def _get_base_pressure_interpolant(self, blade_type: Literal['conv', 'conv-div']):
        data_folder = Path(__file__).parents[3] / 'data'
        xq = np.load(data_folder / f'sieverding_{blade_type}_xq.npy')
        yq = np.load(data_folder / f'sieverding_{blade_type}_yq.npy')
        zq = np.load(data_folder / f'sieverding_{blade_type}_zq.npy')
        return make_casadi_interpolant(xq, yq, zq, 'base_pr', 'linear')

    def residual(self, tot_p0, stc_p1, oth_p_base1):
        # Detect array shapes
        num_span = max(tot_p0.shape)

        # `map` makes it a multi-dimensional function
        base_p_interpolant = self._get_base_pressure_interpolant('conv').map(num_span)

        # TODO: Double check this
        # Hardcoded blade parameter (=2)-> Check meaning
        # Add support for other blade parameters

        first_param = stc_p1 / tot_p0
        second_param = 2 * (stc_p1**0)  # it's just a 2 with the correct shape
        table_entry = cs.horzcat(first_param, second_param).T
        pb_by__ptin = base_p_interpolant(table_entry).T
        return oth_p_base1 - pb_by__ptin * tot_p0


class MixingMomentumBalances(EquationBase):
    """
    Balances of mass, momentum and energy for a mixing
    0 = Throat
    1 = Mixed out conditions
    """

    manual_units = ('dimensionless', 'dimensionless', 'J / kg / K')

    scaling_factor = (None, None, 0.01)

    def residual(
        self,
        oth_ch_massflow0,
        # Thermo
        stc_p0,
        stc_p1,
        stc_rhomass0,
        # Kinematics
        kin_W0,
        kin_W1,
        kin_beta0,
        kin_beta1,
        kin_relmach0,
        kin_relmach1,
        # Geometry
        geo_hh0,
        geo_metal_angle0,
        geo_pitch0,
        geo_pitch1,
        geo_bld_thick0,
        # Boundary layer
        oth_p_base0,
        oth_mom_thick0,
        # Entropy production check
        stc_smass0,
        stc_smass1,
        oth_delta_smass_mixing1,
    ):
        # Massflow per unit length
        mf = oth_ch_massflow0 / geo_hh0
        devtn = kin_beta1 - kin_beta0

        # 1 *** X-Momentum
        mom_in_x = (
            mf * kin_W0
            - stc_rhomass0 * kin_W0**2 * oth_mom_thick0
            + stc_p0 * (geo_pitch0 * np.cos(geo_metal_angle0) - geo_bld_thick0)
            + oth_p_base0 * geo_bld_thick0
        )
        mom_out_x = mf * kin_W1 * np.cos(devtn) + stc_p1 * geo_pitch1 * np.cos(
            geo_metal_angle0
        )
        r_momx = (mom_in_x - mom_out_x) / mom_in_x

        # 2 *** Y-Momentum
        p_suct = stc_p0
        area_y = safe_abs(geo_pitch0 * np.sin(geo_metal_angle0))
        mom_in_y = p_suct * area_y
        mom_out_y = stc_p1 * area_y + mf * kin_W1 * np.sin(devtn)
        r_momy = (mom_in_y - mom_out_y) / mom_in_y

        # 3 *** Supersonic vs. subsonic switch
        switch_supers = kin_relmach0 - 1.0
        switch_subson = kin_beta0 - kin_beta1
        r_regime = cs.if_else(kin_relmach1 >= 0.9, switch_supers, switch_subson)

        # Entropy production for bounding
        r_delta = oth_delta_smass_mixing1 - (stc_smass1 - stc_smass0)

        return r_momx, r_regime, r_delta


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
        zeta = inc_mixing_zeta(
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


def inc_mixing_zeta(
    q,
    stc_p0,
    geo_metal_angle0,
    geo_pitch0,
    geo_bld_thick0,
    oth_p_base0,
    oth_mom_thick0,
    oth_disp_thick0,
):
    w = geo_pitch0 * np.cos(geo_metal_angle0)  # Throat
    cpb = (oth_p_base0 - stc_p0) / q

    return (
        -(cpb * geo_bld_thick0) / w
        + 2 * oth_mom_thick0 / w
        + ((oth_disp_thick0 + geo_bld_thick0) / w) ** 2
    )
