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

    manual_units = ('kg / s / m', 'N / m', 'N / m', 'J / kg', 'Pa')

    def _get_base_pressure_interpolant(self, blade_type: Literal['conv', 'conv-div']):
        data_folder = Path(__file__).parents[3] / 'data'
        xq = np.load(data_folder / f'sieverding_{blade_type}_xq.npy')
        yq = np.load(data_folder / f'sieverding_{blade_type}_yq.npy')
        zq = np.load(data_folder / f'sieverding_{blade_type}_zq.npy')
        return make_casadi_interpolant(xq, yq, zq, 'base_pr', 'linear')

    def residual(
        self,
        # Thermo
        stc_p0,
        stc_p1,
        stc_rhomass0,
        stc_rhomass1,
        tot_p0,  # For sieverding
        tot_hmass0,
        tot_hmass1,
        # Kinematics
        kin_W0,
        kin_W1,
        kin_metal_angle0,
        kin_beta0,
        kin_beta1,
        # Geometry
        geo_pitch0,
        geo_bld_thick0,
        # Boundary layer
        oth_p_base0,
        oth_disp_thick0,
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
        second_param = 2 * stc_p1**0
        table_entry = cs.horzcat(first_param, second_param)
        r4 = oth_p_base0 - base_p_interpolant(table_entry.T).T

        # NOTE:
        # 1. These balances are on a control volume with inlet and outlet
        # perpendicular to the relative velocity (not meridional)
        # 2. The sign of deviation should not be relevant
        # because the cosine makes it positive anyways
        # 3. Use metal angle instead of beta?
        alpha = kin_metal_angle0
        w = geo_pitch0 * np.cos(alpha)
        delta = alpha - kin_beta1
        p_suc = stc_p0

        mass_in = stc_rhomass0 * kin_W0 * (w - geo_bld_thick0 - oth_disp_thick0)
        mass_out = stc_rhomass1 * kin_W1 * geo_pitch0 * np.cos(alpha - delta)

        r0 = mass_in - mass_out

        # Momentum in x direction
        mom_in_x = (
            stc_p0 * (w - geo_bld_thick0)  # Throat minus te thickness
            + oth_p_base0 * geo_bld_thick0  # Base pressure contribution
            + mass_in * kin_W0  # Incoming massflow
            - stc_rhomass0 * kin_W0**2 * oth_mom_thick0  # Deficit due to b.l.
        )
        mom_out_x = mass_out * kin_W1 * np.cos(delta) + stc_p1 * w
        r1 = mom_in_x - mom_out_x

        # Momentum in y direction
        r2 = (p_suc - stc_p1) * w * np.tan(alpha) - mass_out * kin_W1**2 * np.sin(delta)

        # Energy balance
        r3 = tot_hmass0 - tot_hmass1

        return r0, r1, r2, r3, r4
