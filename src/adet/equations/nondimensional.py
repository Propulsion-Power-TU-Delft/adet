"""
Module that gathers equations that represent definitions of nondimensional
coefficients used in TurboMachinery
"""

from adet.equations import EquationBase
import numpy as np
import casadi as cs
import CoolProp as cp


class TotalTotalPressureRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def residual(self, tot_p0, tot_p1, oth_pRatio_tt1):
        return tot_p0 * oth_pRatio_tt1 - tot_p1


class StaticTotalPressRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def residual(self, tot_p0, stc_p1, oth_pRatio_ts1):
        return tot_p0 * oth_pRatio_ts1 - stc_p1


class MidspanTotalTotalPressRatio(EquationBase):
    def residual(self, tot_p0, tot_p1, oth_pRatio_tt_midspan1):
        num_span = max(tot_p0.shape)

        if num_span == 1:
            midspan = 0
        else:
            midspan = num_span // 2

        return tot_p0[midspan] * oth_pRatio_tt_midspan1 - tot_p1[midspan]


class IsentropicTotalEnthalpy(EquationBase):
    input_pair = cp.PSmass_INPUTS
    output_quantities = ('hmass',)
    manual_units = ('J / kg',)

    def residual(self, oth_tot_hmass_is1, stc_smass0, tot_p1):
        return oth_tot_hmass_is1 - self.eos(tot_p1, stc_smass0)


class TotalTotalCompressionEfficiency(EquationBase):
    def residual(
        self,
        tot_p1,
        stc_smass0,
        tot_hmass0,
        tot_hmass1,
        oth_eta_tt1,
        oth_tot_hmass_is1,
    ):
        return oth_eta_tt1 - (oth_tot_hmass_is1 - tot_hmass0) / (
            tot_hmass1 - tot_hmass0
        )


class StaticStaticPressRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def residual(self, stc_p0, stc_p1, oth_pRatio_ss1):
        return stc_p0 * oth_pRatio_ss1 - stc_p1


class DensityRatio(EquationBase):
    """
    .. math::
        \\mathrm{FR} = \\frac{\\rho_{t1}}{p_{t,0}}
    """

    def residual(self, tot_p0, stc_p1, oth_rhoRatio1):
        return oth_rhoRatio1 - stc_p1 / tot_p0


class FlowCoefficient(EquationBase):
    """
    .. math::
        \\phi = \\frac{V_{m0}}{U_{0}}
    """

    def residual(self, kin_Vm0, kin_U0, oth_flowCoeff0):
        return kin_U0 * oth_flowCoeff0 - kin_Vm0


class WorkCoefficient(EquationBase):
    """
    .. math::
        \\psi = \\frac{L_{eul}}{U_0V_{t0}}

    Note
    ----
    In some literature the denominator is :math:`2U_0V_{t0}`
    """

    def residual(self, tot_hmass0, tot_hmass1, kin_Vt0, kin_U0, oth_workCoeff1):
        return (tot_hmass0 - tot_hmass1) - kin_U0**2 * oth_workCoeff1


class SwallowingCapacity(EquationBase):
    """
    .. math::
        \\phi_{t0} = \\frac{\\dot{m}}{\\rho_{t0}A_1U_1}


    Taken from pg 254 Casey Turbocompressors
    """

    def residual(self, oth_massflow0, tot_rhomass0, geo_area1, kin_U1, oth_swllCap1):
        return oth_swllCap1 - oth_massflow0 / (tot_rhomass0 * geo_area1 * kin_U1)


class SpecificSpeed(EquationBase):
    def residual(
        self,
        oth_specificSpeed1,
        kin_omega1,
        oth_massflow1,
        stc_rhomass1,
        tot_hmass0,
        stc_hmass1,
    ):
        return oth_specificSpeed1 * (
            (tot_hmass0 - stc_hmass1) ** (3 / 4)
        ) - kin_omega1 * np.sqrt(oth_massflow1 / stc_rhomass1)


class SizeParameter(EquationBase):
    def residual(
        self, oth_sizeParameter1, oth_massflow1, stc_rhomass1, tot_hmass0, stc_hmass1
    ):
        return oth_sizeParameter1 * ((tot_hmass0 - stc_hmass1) ** (1 / 4)) - np.sqrt(
            oth_massflow1 / stc_rhomass1
        )


class AbsoluteMachNumber(EquationBase):
    def residual(self, kin_mach0, kin_V0, stc_speed_sound0):
        return kin_mach0 * stc_speed_sound0 - kin_V0


class RelativeMachNumber(EquationBase):
    manual_units = ('dimensionless',)

    def residual(self, kin_relmach0, kin_W0, stc_speed_sound0):
        # Choking criterion
        choke = cs.if_else(kin_relmach0 > 1.0, 1.0, kin_relmach0)
        return kin_relmach0 * stc_speed_sound0 - kin_W0
