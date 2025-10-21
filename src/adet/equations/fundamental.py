"""
Module that gathers fundamental equations for internal flows
"""

from adet.equations import EquationBase
import numpy as np


class MassConservation(EquationBase):
    def residual(self, oth_massflow0, oth_massflow1):
        return oth_massflow0 - oth_massflow1


class EulerEquation(EquationBase):
    def residual(self, tot_hmass0, kin_U0, kin_Vt0, tot_hmass1, kin_U1, kin_Vt1):
        return (tot_hmass1 - tot_hmass0) - (kin_U1 * kin_Vt1 - kin_U0 * kin_Vt0)


class CumMassFlow(EquationBase):
    """
    Cumulative massflow
    """

    def residual(self, oth_cum_massflow0, oth_massflow0):
        return oth_cum_massflow0 - np.sum(oth_massflow0)


class MassAreaRelation(EquationBase):
    """
    .. math::
        \\dot_{m} = \\rho_0 V_{m0} A_0
    """

    def residual(self, kin_Vm0, kin_area0, stc_rhomass0, oth_massflow0):
        return oth_massflow0 - stc_rhomass0 * kin_Vm0 * kin_area0


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
    ):
        r1 = tot_hmass0 - (stc_hmass0 + kin_V0**2 / 2)
        r2 = rlt_hmass0 - (stc_hmass0 + kin_W0**2 / 2)
        r3 = tot_smass0 - stc_smass0
        r4 = rlt_smass0 - stc_smass0

        return r1, r2, r3, r4


class FreeVortexDistribution(EquationBase):
    def residual(self, kin_rr0, kin_Vt0, kin_rmid0, oth_Vtmid0):
        return kin_rr0 * kin_Vt0 - kin_rmid0 * oth_Vtmid0


class ForcedVortexDistribution(EquationBase):
    def residual(self, kin_rr0, kin_Vt0, kin_rmid0, oth_Vtmid0):
        return kin_Vt0 / kin_rr0 - oth_Vtmid0 / kin_rmid0


class GeneralWhirl(EquationBase):
    def residual(self, kin_rr0, kin_Vt0, gen_whirl_a, gen_whirl_b, gen_whirl_n):
        return kin_Vt0 - gen_whirl_a * kin_rr0**gen_whirl_n + gen_whirl_b / kin_rr0


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
    ):
        r1 = kin_V0**2 - (kin_Vm0**2 + kin_Vt0**2)
        r2 = kin_W0**2 - (kin_Wm0**2 + kin_Wt0**2)
        r3 = kin_Vm0 - kin_Wm0
        r4 = kin_Vt0 - (kin_Wt0 + kin_U0)
        r5 = kin_alpha0 - np.atan2(kin_Vt0, kin_Vm0)
        r6 = kin_beta0 - np.atan2(kin_Wt0, kin_Wm0)

        return r1, r2, r3, r4, r5, r6


class MeridionalUniform(EquationBase):
    # = * = * = * = * = * = * = * = * = * = * = * = * = * = *
    # * BOUNTY (One or multiple beers):                                   =
    # = Add differential equation for streamline curvature  *
    # * instead of uniform distribution                     =
    # = * = * = * = * = * = * = * = * = * = * = * = * = * = *

    def residual(
        self,
        kin_rr0,
        kin_rmid0,
        kin_height0,
        kin_hh0,
        kin_meridional_angle0,
        kin_area0,
    ):
        spanwise_stations = max(kin_rr0.shape)
        if spanwise_stations == 1:
            r1 = kin_rr0 - kin_rmid0
            r2 = kin_hh0 - kin_height0
        else:
            unit_space = np.linspace(0, 1, spanwise_stations)

            r1 = kin_rr0 - (
                kin_rmid0[0]
                - kin_height0[0] / 2 * np.cos(kin_meridional_angle0[0])
                + unit_space * (kin_height0[0] * np.cos(kin_meridional_angle0[0]))
            )

            r2 = kin_hh0 - kin_height0 / spanwise_stations

        r3 = kin_area0 - np.pi * (
            (kin_rr0 + kin_hh0 / 2) ** 2 - (kin_rr0 - kin_hh0 / 2) ** 2
        )

        return r1, r2, r3
