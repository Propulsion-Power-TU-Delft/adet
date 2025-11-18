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
        geo_pitch0,
        geo_bld_thick0,
        # Boundary layer
        oth_disp_thick0,
        oth_mom_thick0,
    ):
        # Detect array shapes
        num_span = max(kin_W0.shape)

        # NOTE:
        # These balances are on a control volume with inlet and outlet
        # perpendicular to the relative velocity (not meridional)
        throat = geo_pitch0 * np.cos(kin_beta0)
        out_passage = geo_pitch0 * np.cos(kin_beta1)

        mass_in = stc_rhomass0 * kin_W0 * (throat - oth_disp_thick0)
        mass_out = stc_rhomass1 * kin_W1 * out_passage

        # `map` makes it a multi-dimensional function
        base_p_interpolant = self._get_base_pressure_interpolant('conv').map(num_span)

        # TODO:
        # Hardcoded blade parameter (=2)-> Check meaning
        # Add support for other blade parameters
        first_param = stc_p1 / tot_p0
        second_param = 2
        table_entry = cs.horzcat(first_param, second_param)
        p_base = base_p_interpolant(table_entry.T)

        # NOTE: The sign of deviation should not be relevant
        # because the cosine makes it positive anyways
        deviation = kin_beta0 - kin_beta1

        # Momentum definitions
        mom_in = (
            mass_in * kin_W0  # Incoming massflow
            - stc_rhomass0 * kin_W0**2 * oth_mom_thick0  # Deficit due to b.l.
            + stc_p0 * (throat - geo_bld_thick0)  # Throat minus te thicknes
            + p_base * geo_bld_thick0 * kin_beta0  # Base pressure contribution
        )
        mom_out = mass_out * kin_W1 * np.cos(deviation) + stc_p1 * out_passage

        # NOTE: Right now the RelativeMach equations imposes a maximum
        # 1.0 outlet Mach Number at the row outlet (throat at the outlet)

        # Mass conservation
        r1 = mass_in - mass_out
        # Momentum balance
        r2 = mom_in - mom_out

        # Energy balance (ISENTROPIC MIXING hypothesis)
        r3 = tot_hmass0 - tot_hmass1
        r4 = stc_smass0 - stc_smass1

        return r1, r2, r3, r4


class MixingGeometry(EquationBase):
    """Geometry over the mixing annulus, unchanged"""

    def residual(
        self,
        geo_rmid0,
        geo_rmid1,
        geo_height0,
        geo_height1,
        geo_meridional_angle0,
        geo_meridional_angle1,
    ):
        r1 = geo_rmid1 - geo_rmid0
        r2 = geo_height1 - geo_height0
        r3 = geo_meridional_angle1 - geo_meridional_angle0

        return r1, r2, r3
