"""
Module that gathers fundamental equations for internal flows
"""

from adet.equations import EquationBase
import numpy as np
import casadi as cs
import sympy as sp
from pint.facets.plain import PlainQuantity


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

    def residual(self, kin_Vm0, geo_area0, stc_rhomass0, oth_massflow0):
        return oth_massflow0 - stc_rhomass0 * kin_Vm0 * geo_area0


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
    def residual(self, geo_rr0, kin_Vt0, geo_rmid0, oth_Vtmid0):
        return geo_rr0 * kin_Vt0 - geo_rmid0 * oth_Vtmid0


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
        geo_rr0,
        geo_rmid0,
        geo_height0,
        geo_hh0,
        geo_meridional_angle0,
        geo_area0,
    ):
        spanwise_stations = max(geo_rr0.shape)
        if spanwise_stations == 1:
            r1 = geo_rr0 - geo_rmid0
            r2 = geo_hh0 - geo_height0
        else:
            unit_space = np.linspace(0, 1, spanwise_stations)

            r1 = geo_rr0 - (
                geo_rmid0[0]
                - geo_height0[0] / 2 * np.cos(geo_meridional_angle0[0])
                + unit_space * (geo_height0[0] * np.cos(geo_meridional_angle0[0]))
            )

            r2 = geo_hh0 - geo_height0 / spanwise_stations

        r3 = geo_area0 - np.pi * (
            (geo_rr0 + geo_hh0 / 2) ** 2 - (geo_rr0 - geo_hh0 / 2) ** 2
        )

        return r1, r2, r3


def is_casady_type(x):
    return isinstance(x, (cs.DM, cs.MX, cs.SX))


def safe_min_clip(x, min_value):
    """
    Lower clipping of the absolute vaue of x
    with respect to a minimum value.
    Type safe for casadi, numpy and pint
    """
    if is_casady_type(x):
        x = cs.fmax(cs.fabs(x), min_value)
    elif isinstance(x, PlainQuantity):
        x = np.clip(np.abs(x.magnitude), min_value * x.units, None)
    elif isinstance(x, sp.Symbol):
        pass
    else:
        x = np.clip(np.abs(x), min_value, None)

    return x


class ParabolicCamberline(EquationBase):
    # NOTE: You can use this and skip the safe checks
    skip_unit_check = True
    manual_units = ('m', 'm', 'rad')

    @staticmethod
    def _compute_parabola(geo_beta0, geo_beta1, chord):
        """
        Compute a, b for y = ax^2 + bx such that:
        dy/dx at x=0 = tan(beta0), at x=L = tan(beta1)
        """
        delta_angle = geo_beta1 - geo_beta0
        tan0 = np.tan(delta_angle / 2)
        tan1 = np.tan(-delta_angle / 2)

        a = (tan1 - tan0) / (2 * chord)
        b = tan0

        a = safe_min_clip(a, 0.001)

        return a, b

    @staticmethod
    def _parabolic_arc_len(a, b, chord):
        """
        Exact arc length of y = ax² + bx from x = 0 to x = chord_ax
        """
        term1 = 2 * a * chord + b
        term0 = b

        sqrt1 = np.sqrt(1 + term1**2)
        sqrt0 = np.sqrt(1 + term0**2)

        asinh1 = np.arcsinh(term1)
        asinh0 = np.arcsinh(term0)

        length = (1 / (4 * a)) * (term1 * sqrt1 + asinh1 - term0 * sqrt0 - asinh0)

        return length

    def residual(
        self,
        geo_beta0,
        geo_beta1,
        geo_chord1,
        geo_stagger1,
        geo_chord_ax1,
        geo_camb_len1,
    ):
        a, b = self._compute_parabola(geo_beta0, geo_beta1, geo_chord1)
        arc_len = self._parabolic_arc_len(a, b, geo_chord1)

        deflection = geo_beta1 - geo_beta0

        r1 = geo_chord_ax1 - geo_chord1 * np.cos(geo_stagger1)
        r2 = geo_camb_len1 - arc_len
        r3 = geo_stagger1 - (deflection / 2 - geo_beta0)
        return r1, r2, r3


if __name__ == '__main__':
    eq = CumMassFlow()
    eq._count_equations()
