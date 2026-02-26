"""Module that gathers fundamental equations for internal flows"""

import numpy as np

from adet.equations import EquationBase
from adet.equations.utils import get_midspan_idx, safe_sum, span_fin_diff
from adet.equations.base_equation import MeridAreaBlockage


class EulerEquation(EquationBase):
    def residual(self, tot_hmass0, kin_U0, kin_Vt0, tot_hmass1, kin_U1, kin_Vt1):
        return (tot_hmass1 - tot_hmass0) - (kin_U1 * kin_Vt1 - kin_U0 * kin_Vt0)


class ConstantAngMomentum(EquationBase):
    def residual(self, geo_rr0, geo_rr1, kin_Vt0, kin_Vt1):
        return geo_rr0 * kin_Vt0 - geo_rr1 * kin_Vt1


class ConstantEnergy(EquationBase):
    def residual(self, rlt_hmass0, rlt_hmass1):
        # WARN: This is used in mixing models for AXIAL
        # turbines, I need to double check this
        return rlt_hmass0 - rlt_hmass1


class MassConservation(EquationBase):
    def residual(self, oth_massflow0, oth_massflow1):
        return oth_massflow0 - oth_massflow1


class TotalMassFlow(EquationBase):
    """Cumulative massflow"""

    def residual(self, oth_cum_massflow0, oth_massflow0):
        return oth_cum_massflow0 - safe_sum(oth_massflow0)


class TotalArea(EquationBase):
    def residual(self, geo_cum_area0, geo_area0):
        return geo_cum_area0 - safe_sum(geo_area0)


class MassAreaRelation(EquationBase):
    """
    .. math::
        \\dot_{m} = \\rho_0 V_{m0} A_0
    """

    def residual(self, kin_Vm0, geo_eff_area0, stc_rhomass0, oth_massflow0):
        return oth_massflow0 - stc_rhomass0 * kin_Vm0 * geo_eff_area0


class ZeroBlockage(MeridAreaBlockage):
    """Use the annuli's area as the passage area"""

    def residual(self, geo_area0, geo_eff_area0):
        return geo_eff_area0 - geo_area0


class BladeBlockage(MeridAreaBlockage):
    def residual(
        self,
        geo_hh0,
        geo_area0,  # Full annulus area
        geo_eff_area0,  # Blocked area
        geo_num_blades0,
        geo_bld_thick0,  # Blade thickness
        geo_metal_angle0,
        oth_disp_thick0,  # COMBINED diplacement thickness
    ):
        return geo_eff_area0 - (
            geo_area0
            - geo_num_blades0
            * geo_hh0
            * (geo_bld_thick0 + oth_disp_thick0)
            / np.cos(geo_metal_angle0)
        )


class Kinematics(EquationBase):
    def residual(
        self,
        kin_V0,
        kin_Vm0,
        kin_Vt0,
        kin_W0,
        kin_Wt0,
        kin_Wm0,
        kin_U0,
        kin_alpha0,
        kin_beta0,
        kin_omega0,
        geo_rr0,
    ):
        # Only if Vm and Vt are zero the denominator
        # nullifies, but Vm > 0 always, thus the
        # square root should pose no problems
        r1 = kin_V0 - (kin_Vm0**2 + kin_Vt0**2) ** 0.5
        r2 = kin_W0 - (kin_Wm0**2 + kin_Wt0**2) ** 0.5

        r3 = kin_Vm0 - kin_Wm0
        r4 = kin_Vt0 - (kin_Wt0 + kin_U0)

        # *** atan2 ensures that the angles are between - pi / 2 and pi / 2
        r5 = kin_alpha0 - np.atan2(kin_Vt0, kin_Vm0)
        r6 = kin_beta0 - np.atan2(kin_Wt0, kin_Wm0)

        # *** OLD Alternative formulation - Can be used in
        # combination with bounds
        # r5 = kin_Wm0 - kin_W0 * np.cos(kin_beta0)
        # r6 = kin_Vm0 - kin_V0 * np.cos(kin_alpha0)

        r7 = kin_omega0 * geo_rr0 - kin_U0

        return r1, r2, r3, r4, r5, r6, r7


class TotalStaticMatching(EquationBase):
    """
    Match the total and static states imposing equal
    entropy and

    .. math::
        h_{t0} = h_0 + \\frac{V_0^2}{2}

    .. math::
        h_{t0}^{rel} = h_0 + \\frac{W_0^2}{2}

    Note
    ----
    The total and relative total entropy do not have a real
    physical sense, as the total and rel. tot. states are defined
    by an isentropic alting of the flow.

    Nonetheless, our formulation treats the
    three states as independent equations of state, which are
    matched in an Equation-Oriented using this equation.

    Observe that mathematically the entropy equality does not affect
    in any way the convergence of the system, and it is immediatly
    satisfied after the first N-R Iteration
    """

    def residual(
        self,
        tot_hmass0,
        stc_hmass0,
        rlt_hmass0,
        tot_smass0,
        stc_smass0,
        rlt_smass0,
        kin_V0,
        kin_W0,
        # Force to add density as variables in all states
    ):
        r1 = tot_hmass0 - (stc_hmass0 + kin_V0**2 / 2)
        r2 = rlt_hmass0 - (stc_hmass0 + kin_W0**2 / 2)
        r3 = tot_smass0 - stc_smass0
        r4 = rlt_smass0 - stc_smass0

        return r1, r2, r3, r4


class SimpleRadialEquilibrium(EquationBase):
    """
    Most implementation of a radial equilibrium,
    zero streamline curvature is assumed
    """

    manual_units = ('J / kg / m',)

    def residual(self, geo_rr0, stc_p0, kin_Vt0, stc_rhomass0):
        dp_dr = span_fin_diff(stc_p0, geo_rr0)
        return dp_dr / stc_rhomass0 - kin_Vt0**2 / geo_rr0


class NisRe(EquationBase):
    """Non-ISentropic Radial Equilibrium"""

    manual_units = ('J / kg / m',)

    def residual(self, geo_rr0, kin_Vt0, kin_Vm0, tot_hmass0, stc_T0, stc_smass0):
        dVt_dr = span_fin_diff(kin_Vt0, geo_rr0)
        dVm_dr = span_fin_diff(kin_Vm0, geo_rr0)
        dht_dr = span_fin_diff(tot_hmass0, geo_rr0)
        ds_dr = span_fin_diff(stc_smass0, geo_rr0)

        lhs = kin_Vm0 * dVm_dr + kin_Vt0 * dVt_dr + kin_Vt0**2 / geo_rr0
        rhs = dht_dr - stc_T0 * ds_dr
        return lhs - rhs


class FreeVortexDistribution(EquationBase):
    def residual(self, geo_rr0, kin_Vt0):
        midspan = get_midspan_idx(geo_rr0)
        rVt_mid = geo_rr0[midspan] * kin_Vt0[midspan]

        r1 = geo_rr0[:midspan] * kin_Vt0[:midspan] - rVt_mid
        r2 = geo_rr0[midspan + 1 :] * kin_Vt0[midspan + 1 :] - rVt_mid
        return r1, r2


class ForcedVortexDistribution(EquationBase):
    def residual(self, geo_rr0, kin_Vt0):
        midspan = get_midspan_idx(geo_rr0)
        Vt_by_r_mid = kin_Vt0[midspan] / geo_rr0[midspan]

        r1 = kin_Vt0[:midspan] / geo_rr0[:midspan] - Vt_by_r_mid
        r2 = kin_Vt0[midspan + 1 :] / geo_rr0[midspan + 1 :] - Vt_by_r_mid
        return r1, r2


class GeneralWhirl(EquationBase):
    def residual(self, geo_rr0, kin_Vt0, gen_whirl_a, gen_whirl_b, gen_whirl_n):
        free_vortex_term = kin_Vt0 * geo_rr0
        frcd_vortex_term = kin_Vt0 / geo_rr0
        return kin_Vt0 - gen_whirl_a * geo_rr0**gen_whirl_n + gen_whirl_b / geo_rr0
