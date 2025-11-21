"""Module that gathers fundamental equations for internal flows"""

import numpy as np

from adet.equations import EquationBase
from adet.tools.interpolation import fin_diff


class EulerEquation(EquationBase):
    def residual(self, tot_hmass0, kin_U0, kin_Vt0, tot_hmass1, kin_U1, kin_Vt1):
        return (tot_hmass1 - tot_hmass0) - (kin_U1 * kin_Vt1 - kin_U0 * kin_Vt0)


class MassConservation(EquationBase):
    def residual(self, oth_massflow0, oth_massflow1):
        return oth_massflow0 - oth_massflow1


class MassAreaRelation(EquationBase):
    """
    .. math::
        \\dot_{m} = \\rho_0 V_{m0} A_0
    """

    def residual(self, kin_Vm0, geo_eff_area0, stc_rhomass0, oth_massflow0):
        return oth_massflow0 - stc_rhomass0 * kin_Vm0 * geo_eff_area0


class ZeroBlockage(EquationBase):
    """Use the annuli's area as the passage area"""

    def residual(self, geo_area0, geo_eff_area0):
        return geo_eff_area0 - geo_area0


class BladeBlockage(EquationBase):
    def residual(
        self,
        geo_hh0,
        geo_area0,
        geo_eff_area0,
        geo_num_blades0,
        geo_bld_thick0,
        geo_metal_angle0,
        # Boudndary Layer
        oth_disp_thick0,
    ):
        return geo_eff_area0 - (
            geo_area0
            - geo_num_blades0
            * geo_hh0
            * (geo_bld_thick0 + 2 * oth_disp_thick0)
            / np.cos(geo_metal_angle0)
        )


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

    skip_unit_check = True
    manual_units = ('J / kg / m',)

    def residual(self, geo_rr0, stc_p0, kin_Vt0, stc_rhomass0):
        dp_dr = fin_diff(stc_p0, geo_rr0)
        return dp_dr / stc_rhomass0 - kin_Vt0**2 / geo_rr0


class NisRe(EquationBase):
    """Non-ISentropic Radial Equilibrium"""

    skip_unit_check = True
    manual_units = ('J / kg / m',)

    def residual(self, geo_rr0, kin_Vt0, kin_Vm0, tot_hmass0, stc_T0, stc_smass0):
        dVt_dr = fin_diff(kin_Vt0, geo_rr0)
        dVm_dr = fin_diff(kin_Vm0, geo_rr0)
        dht_dr = fin_diff(tot_hmass0, geo_rr0)
        ds_dr = fin_diff(stc_smass0, geo_rr0)

        lhs = kin_Vm0 * dVm_dr + kin_Vt0 * dVt_dr + kin_Vt0**2 / geo_rr0
        rhs = dht_dr - stc_T0 * ds_dr
        return lhs - rhs


class FreeVortexDistribution(EquationBase):
    def residual(self, geo_rr0, kin_Vt0, geo_rmid0, oth_Vt_mid0):
        return geo_rr0 * kin_Vt0 - geo_rmid0 * oth_Vt_mid0


class ForcedVortexDistribution(EquationBase):
    def residual(self, geo_rr0, kin_Vt0, geo_rmid0, oth_Vtmid0):
        return kin_Vt0 / geo_rr0 - oth_Vtmid0 / geo_rmid0


class GeneralWhirl(EquationBase):
    def residual(self, geo_rr0, kin_Vt0, gen_whirl_a, gen_whirl_b, gen_whirl_n):
        return kin_Vt0 - gen_whirl_a * geo_rr0**gen_whirl_n + gen_whirl_b / geo_rr0


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

        # atan2 ensures that the angles are between - pi / 2 and pi / 2
        r5 = kin_alpha0 - np.atan2(kin_Vt0, kin_Vm0)
        r6 = kin_beta0 - np.atan2(kin_Wt0, kin_Wm0)
        # - Alternative formulation
        # r5 = kin_Wm0 - kin_W0 * np.cos(kin_beta0)
        # r6 = kin_Vm0 - kin_V0 * np.cos(kin_alpha0)

        # Glad to have you back here peripheral velocity relation.
        # Sorry for your brief adventure outside of Kinematics,
        # you really do belong here.
        r7 = kin_omega0 * geo_rr0 - kin_U0

        return r1, r2, r3, r4, r5, r6, r7
