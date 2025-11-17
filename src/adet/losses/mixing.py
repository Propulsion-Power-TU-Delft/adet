"""Mixing losses downstream of turbomachinery blades"""

from pathlib import Path
from typing import Literal
import casadi as cs
import numpy as np
from adet.equations.base_equation import EquationBase
from adet.tools.interpolation import make_casadi_interpolant


class MixingBalances(EquationBase):
    """
    Balances of mass, momentum and energy for a mixing
    0 = Throat
    1 = Mixed out conditions
    """

    skip_unit_check = True
    manual_units = ('kg / s / m', 'N / m', 'J / kg', 'J / kg / K')

    def _get_base_pressure_interpolant(self, blade_type: Literal['conv', 'conv-div']):
        data_folder = Path(__file__).parents[3] / 'data'
        xq = np.load(data_folder / f'sieverding_{blade_type}_xq.npy')
        yq = np.load(data_folder / f'sieverding_{blade_type}_yq.npy')
        zq = np.load(data_folder / f'sieverding_{blade_type}_zq.npy')
        return make_casadi_interpolant(xq, yq, zq, 'base_pr', 'linear')

    def residual(
        self,
        # Thermo
        stc_speed_sound0,
        stc_rhomass0,
        stc_rhomass1,
        stc_p0,
        stc_p1,
        tot_p0,
        tot_hmass0,
        tot_hmass1,
        stc_smass0,
        stc_smass1,
        # Kine
        kin_W0,
        kin_W1,
        kin_beta0,
        kin_beta1,
        # Geometry
        geo_throat0,
        geo_pitch0,
        geo_te_thick0,
        # Boundary layer
        oth_disp_thick0,
        oth_mom_thick0,
        oth_massflow0,
    ):
        # Detect array shapes
        num_span = max(kin_W0.shape)

        # NOTE:
        # These balances are on a control volume with inlet and outlet
        # perpendicular to the relative velocity (not meridional)
        mass_in = stc_rhomass0 * kin_W0 * (geo_throat0 - oth_disp_thick0)
        mass_out = stc_rhomass1 * kin_W1 * np.cos(kin_beta1) * geo_pitch0

        # `map` makes it a multi-dimensional function
        base_p_interpolant = self._get_base_pressure_interpolant('conv').map(num_span)

        # TODO:
        # 1. Hardcoded blade parameter -> Check meaning
        # 2. Add support for other blade parameters
        first_param = stc_p1 / tot_p0
        second_param = 2
        table_entry = cs.horzcat(first_param, second_param)
        p_base = base_p_interpolant(table_entry.T)

        # NOTE: The sign of deviation should not be relevant
        # because the cosine makes it positive anyways
        # Note to self, if numerical problems => Add cs.fabs
        deviation = kin_beta0 - kin_beta1

        mom_in = (
            mass_in * kin_W0
            - stc_rhomass0 * kin_W0**2 * oth_disp_thick0
            + stc_p0 * geo_throat0
            + p_base * geo_te_thick0 * kin_beta0
        )
        mom_out = mass_out * kin_W1 * np.cos(deviation) + stc_p1 * geo_pitch0 * np.cos(
            kin_beta1
        )

        # NOTE: Right now the RelativeMach equations imposes a maximum
        # 1.0 outlet Mach Number at the row outlet (throat at the outlet)

        # Mass conservation
        r1 = mass_in - mass_out
        # Momentum balance
        r2 = mom_in - mom_out
        # Energy balance + Isentropic mixing
        r3 = tot_hmass0 - tot_hmass1
        r4 = stc_smass0 - stc_smass1

        return r1, r2, r3, r4


class MixingGeometry(EquationBase):
    def residual(
        self,
        geo_height0,
        geo_rmid0,
        geo_meridional_angle0,
        geo_height1,
        geo_rmid1,
        geo_meridional_angle1,
        geo_throat0,
    ):
        r1 = geo_height1 - geo_height0
        r2 = geo_rmid1 - geo_rmid0
        r3 = geo_meridional_angle1 - geo_meridional_angle0

        return r1, r2, r3


class BoundaryLayerProperties(EquationBase):
    """Boundary layer properties ratios based on trailing edge thickness"""

    def residual(
        self,
        # Geometry
        geo_pitch0,
        geo_te_thick0,
        geo_te_by_pitch0,
        # Boundary layer
        oth_disp_thick0,
        oth_disp_by_mom_thick0,
        oth_mom_thick0,
        oth_mom_by_te_thick0,
    ):
        r1 = oth_disp_thick0 - oth_disp_by_mom_thick0 * oth_mom_thick0
        r2 = oth_mom_thick0 - oth_mom_by_te_thick0 * geo_te_thick0

        return r1, r2
