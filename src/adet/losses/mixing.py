"""Mixing losses downstream of turbomachinery blades"""

from adet.equations.utils import safe_abs

from pathlib import Path
from typing import Literal
import casadi as cs
import numpy as np
from adet.equations.base_equation import EquationBase
from adet.tools.interpolation import make_casadi_interpolant


class MixingMomentumBalances(EquationBase):
    """
    Balances of mass, momentum and energy for a mixing
    0 = Throat
    1 = Mixed out conditions
    """

    manual_units = ('Pa', 'N / m', 'N / m', 'J / kg / K')

    def _get_base_pressure_interpolant(self, blade_type: Literal['conv', 'conv-div']):
        data_folder = Path(__file__).parents[3] / 'data'
        xq = np.load(data_folder / f'sieverding_{blade_type}_xq.npy')
        yq = np.load(data_folder / f'sieverding_{blade_type}_yq.npy')
        zq = np.load(data_folder / f'sieverding_{blade_type}_zq.npy')
        return make_casadi_interpolant(xq, yq, zq, 'base_pr', 'linear')

    def residual(
        self,
        oth_ch_massflow0,
        # Thermo
        stc_p0,
        stc_p1,
        stc_rhomass0,
        tot_p0,  # For sieverding
        # Kinematics
        kin_W0,
        kin_W1,
        kin_beta0,
        kin_beta1,
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
        delta_smass_mixing1,
    ):
        # Detect array shapes
        num_span = max(kin_W0.shape)

        # `map` makes it a multi-dimensional function
        base_p_interpolant = self._get_base_pressure_interpolant('conv').map(num_span)

        # TODO: Double check this
        # Hardcoded blade parameter (=2)-> Check meaning
        # Add support for other blade parameters
        first_param = stc_p1 / tot_p0
        second_param = 2 * stc_p1**0  # it's just a 2 with the correct shape
        table_entry = cs.horzcat(first_param, second_param).T
        pb_by__ptin = base_p_interpolant(table_entry).T
        r0 = oth_p_base0 - pb_by__ptin * tot_p0

        # Hypotheses
        p_suct = (stc_p0 + stc_p1) / 2
        devtn = kin_beta1 - kin_beta0

        # Massflow per unit length
        mf = oth_ch_massflow0 / geo_hh0

        # X-Momentum
        mom_in_x = (
            mf * kin_W0
            - stc_rhomass0 * kin_W0**2 * oth_mom_thick0
            + stc_p0 * (geo_pitch0 * np.cos(geo_metal_angle0) - geo_bld_thick0)
            + oth_p_base0 * geo_bld_thick0
        )
        mom_out_x = mf * kin_W1 * np.cos(devtn) + stc_p1 * geo_pitch1 * np.cos(
            geo_metal_angle0
        )
        r1 = mom_in_x - mom_out_x

        # Y-Momentum
        area_y = safe_abs(geo_pitch0 * np.sin(geo_metal_angle0))
        mom_in_y = p_suct * area_y
        mom_out_y = stc_p1 * area_y + mf * kin_W1 * np.sin(
            -cs.sign(geo_metal_angle0) * devtn
        )
        r2 = mom_in_y - mom_out_y

        # Entropy production for bounding
        r3 = delta_smass_mixing1 - (stc_smass1 - stc_smass0)

        return r0, r1, r2, r3
