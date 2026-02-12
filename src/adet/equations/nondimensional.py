"""
Module that gathers equations that represent definitions of nondimensional
coefficients used in TurboMachinery
"""

import numpy as np
import casadi as cs
import CoolProp as cp

from adet.equations import EquationBase
from adet.equations.utils import safe_abs, thermo_deriv


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


class TotalTotalExpansionEfficiency(EquationBase):
    manual_units = ('dimensionless',)
    input_pair = cp.PSmass_INPUTS
    output_quantities = ('hmass',)

    def residual(
        self,
        stc_smass0,
        tot_p1,
        tot_hmass0,
        tot_hmass1,
        oth_eta_tt1,
    ):
        tot_hmass_is1 = self.eos(tot_p1, stc_smass0)
        return oth_eta_tt1 - (tot_hmass0 - tot_hmass1) / (tot_hmass0 - tot_hmass_is1)


class TotalTotalCompressionEfficiency(EquationBase):
    manual_units = ('dimensionless',)
    input_pair = cp.PSmass_INPUTS
    output_quantities = ('hmass',)

    def residual(
        self,
        stc_smass0,
        tot_p1,
        tot_hmass0,
        tot_hmass1,
        oth_eta_tt1,
    ):
        tot_hmass_is1 = self.eos(tot_p1, stc_smass0)
        return oth_eta_tt1 - (tot_hmass_is1 - tot_hmass0) / (tot_hmass1 - tot_hmass0)


class TotalStaticLoadingCoefficient(EquationBase):
    manual_units = ('J / kg',)
    input_pair = cp.PSmass_INPUTS
    output_quantities = ('hmass',)

    def residual(self, oth_ts_loadCoeff1, stc_p1, stc_smass0, tot_hmass0, kin_U1):
        stc_hmass_is1 = self.eos(stc_p1, stc_smass0)
        return kin_U1**2 * oth_ts_loadCoeff1 - 2 * (tot_hmass0 - stc_hmass_is1)


class StaticStaticPressRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def residual(self, stc_p0, stc_p1, oth_pRatio_ss1):
        return stc_p0 * oth_pRatio_ss1 - stc_p1


class TotalStaticDegreeOfReaction(EquationBase):
    """
    0 - [Stator] - 1 === 2 - [Rotor] - 3
    This assumes the stator is on nodes 0,1 and the stator on 2,3 is the rotor.
    The degree of reaction is an `oth` property of node 3
    """

    def residual(
        self,
        tot_hmass0,
        stc_hmass1,
        stc_hmass2,
        stc_hmass3,
        tot_hmass3,
        oth_reactDegree_ts3,
    ):
        delta_hmass_rotor = stc_hmass3 - stc_hmass2
        delta_tot_hmass_stage = tot_hmass3 - tot_hmass0

        return delta_tot_hmass_stage * oth_reactDegree_ts3 - delta_hmass_rotor


class StaticDegreeOfReaction(EquationBase):
    """
    0 - [Stator] - 1 === 2 - [Rotor] - 3
    This assumes the stator is on nodes 0,1 and the stator on 2,3 is the rotor.
    The degree of reaction is an `oth` property of node 3
    """

    def residual(
        self,
        stc_hmass0,
        stc_hmass1,
        stc_hmass2,
        stc_hmass3,
        oth_reactDegree3,
    ):
        delta_hmass_rotor = stc_hmass3 - stc_hmass2
        delta_hmass_stage = stc_hmass3 - stc_hmass0

        return delta_hmass_stage * oth_reactDegree3 - delta_hmass_rotor


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

    def residual(self, kin_Vm0, kin_U1, oth_flowCoeff1):
        return safe_abs(kin_U1) * oth_flowCoeff1 - kin_Vm0


class WorkCoefficient(EquationBase):
    """
    .. math::
        \\psi = \\frac{\\Delta h_t}{U_1^2}

    Note
    ----
    In some literature the denominator is :math:`2U_0V_{t0}`
    """

    def residual(self, tot_hmass0, tot_hmass1, kin_U1, oth_workCoeff1):
        return kin_U1**2 * oth_workCoeff1 - (tot_hmass1 - tot_hmass0)


class SwallowingCapacity(EquationBase):
    """
    .. math::
        \\phi_{t0} = \\frac{\\dot{m}}{\\rho_{t0} D_1^2 U_1}


    pg 254 Casey - Radial Flow Turbocompressors
    """

    def residual(
        self,
        kin_U1,
        geo_rr1,
        tot_rhomass0,
        oth_swllCap0,
        oth_massflow0,
    ):
        return oth_swllCap0 - oth_massflow0 / (
            tot_rhomass0 * kin_U1 * (2 * geo_rr1) ** 2
        )


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
        self,
        oth_sizeParameter1,
        oth_massflow1,
        stc_rhomass1,
        tot_hmass0,
        stc_hmass1,
    ):
        return (
            oth_sizeParameter1 * ((tot_hmass0 - stc_hmass1) ** (1 / 4))
            - (oth_massflow1 / stc_rhomass1) ** 0.5
        )


class AbsoluteMachNumber(EquationBase):
    def residual(self, kin_mach0, kin_mermach0, kin_Vm0, kin_V0, stc_speed_sound0):
        r1 = kin_mach0 * stc_speed_sound0 - kin_V0
        r2 = kin_mermach0 * stc_speed_sound0 - kin_Vm0
        return r1, r2


class RelativeMachNumber(EquationBase):
    manual_units = ('dimensionless',)

    def residual(self, kin_relmach0, kin_W0, stc_speed_sound0):
        # Choking criterion, not used for now
        return kin_relmach0 * stc_speed_sound0 - kin_W0


class RelativeMachWithChoke(EquationBase):
    manual_units = ('dimensionless',)

    def residual(self, kin_relmach0, kin_W0, stc_speed_sound0):
        mach_w_choke = cs.if_else(kin_relmach0 > 1.0, 1.0, kin_relmach0)
        return kin_W0 / stc_speed_sound0 - mach_w_choke


class GammaPV(EquationBase):
    # WARN: This update pair for ideal gas
    # Not that using gamma_pv makes sense, but
    # beware
    input_pair = cp.DmassSmass_INPUTS
    output_quantities = ('p',)
    manual_units = ('dimensionless',)

    def residual(self, oth_gamma_pv0, stc_rhomass0, stc_smass0, stc_p0):
        dp_drho = thermo_deriv(self.eos, stc_rhomass0, stc_smass0, 0)[0]
        return oth_gamma_pv0 - stc_rhomass0 / stc_p0 * dp_drho
