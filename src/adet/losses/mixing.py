"""Mixing losses downstream of turbomachinery blades"""

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

    manual_units = ('Pa', 'N / m', 'N / m')

    def _get_base_pressure_interpolant(self, blade_type: Literal['conv', 'conv-div']):
        data_folder = Path(__file__).parents[3] / 'data'
        xq = np.load(data_folder / f'sieverding_{blade_type}_xq.npy')
        yq = np.load(data_folder / f'sieverding_{blade_type}_yq.npy')
        zq = np.load(data_folder / f'sieverding_{blade_type}_zq.npy')
        return make_casadi_interpolant(xq, yq, zq, 'base_pr', 'linear')

    def residual(
        self,
        oth_ch_massflow0,
        oth_ch_massflow1,
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
        geo_hh1,
        geo_pitch0,
        geo_bld_thick0,
        # Boundary layer
        oth_p_base0,
        oth_mom_thick0,
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
        alpha = kin_beta0  # Trailing edge aligned to local flow
        p_suct = stc_p0  # Pressure on suction surface = outlet pressure
        throat = geo_pitch0 * np.cos(alpha)
        devtn = kin_beta1 - alpha

        # Momentum in x direction
        mom_in_x = (
            stc_p0 * (throat - geo_bld_thick0)  # Throat minus te thickness
            + oth_p_base0 * geo_bld_thick0  # Base pressure contribution
            + oth_ch_massflow0 / geo_hh0 * kin_W0  # Incoming massflow
            - stc_rhomass0 * kin_W0**2 * oth_mom_thick0  # Deficit due to b.l.
        )
        mom_out_x = (
            oth_ch_massflow1 / geo_hh1 * kin_W1 * np.cos(devtn) + stc_p1 * throat
        )
        r1 = mom_in_x - mom_out_x

        # Momentum in y direction
        r2 = (p_suct - stc_p1) * throat * np.tan(
            alpha
        ) - oth_ch_massflow1 / geo_hh1 * kin_W1 * np.sin(devtn)

        return r0, r1, r2
