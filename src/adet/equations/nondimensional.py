"""
Module that gathers equations that represent definitions of nondimensional
coefficients used in TurboMachinery
"""

from adet.equations import EquationBase
import numpy as np


class TotalTotalPressureRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def _compute_residual(self, tot_p0, tot_p1, oth_TTratio1):
        return oth_TTratio1 - tot_p1 / tot_p0


class TotalStaticPressureRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def _compute_residual(self, tot_p0, stc_p1, oth_TSratio1):
        return tot_p0 * oth_TSratio1 - stc_p1


class DensityRatio(EquationBase):
    """
    .. math::
        \\mathrm{FR} = \\frac{\\rho_{t1}}{p_{t,0}}
    """

    def _compute_residual(self, tot_p0, stc_p1, oth_rhoRatio1):
        return oth_rhoRatio1 - stc_p1 / tot_p0


class FlowCoefficient(EquationBase):
    """
    .. math::
        \\phi = \\frac{V_{m0}}{U_{0}}
    """

    def _compute_residual(self, kin_Vm0, kin_U0, oth_flowCoeff0):
        return kin_U0 * oth_flowCoeff0 - kin_Vm0


class WorkCoefficient(EquationBase):
    """
    .. math::
        \\psi = \\frac{L_{eul}}{U_0V_{t0}}

    Note
    ----
    In some literature the denominator is :math:`2U_0V_{t0}`
    """

    def _compute_residual(
        self, tot_hmass0, tot_hmass1, kin_Vt0, kin_U0, oth_workCoeff1
    ):
        return kin_U0**2 * oth_workCoeff1 - (tot_hmass0 - tot_hmass1)


class SwallowingCapacity(EquationBase):
    """
    .. math::
        \\phi_{t0} = \\frac{\\dot{m}}{\\rho_{t0}A_1U_1}


    Taken from pg 254 Casey Turbocompressors
    """

    def _compute_residual(
        self, oth_massflow0, tot_rhomass0, kin_area1, kin_U1, oth_swllCap1
    ):
        return oth_swllCap1 - oth_massflow0 / (tot_rhomass0 * kin_area1 * kin_U1)


class SpecificSpeed(EquationBase):
    def _compute_residual(
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
    def _compute_residual(
        self, oth_sizeParameter1, oth_massflow1, stc_rhomass1, tot_hmass0, stc_hmass1
    ):
        return oth_sizeParameter1 * ((tot_hmass0 - stc_hmass1) ** (1 / 4)) - np.sqrt(
            oth_massflow1 / stc_rhomass1
        )


class RadiusRatio(EquationBase):
    def _compute_residual(self, kin_rmid0, kin_rmid1, oth_radiusRatio1):
        return oth_radiusRatio1 - kin_rmid1 / kin_rmid0


class HeightRatio(EquationBase):
    def _compute_residual(self, kin_height0, kin_height1, oth_heightRatio1):
        return oth_heightRatio1 - kin_height1 / kin_height0


class MeridionalVelocityRatio(EquationBase):
    def _compute_residual(self, kin_Vm0, kin_Vm1, oth_VmRatio1):
        return oth_VmRatio1 - kin_Vm1 / kin_Vm0


class AbsoluteMachNumber(EquationBase):
    def _compute_residual(self, oth_mach0, kin_V0, stc_speed_sound0):
        return oth_mach0 * stc_speed_sound0 - kin_V0


class RelativeMachNumber(EquationBase):
    def _compute_residual(self, oth_relmach0, kin_W0, stc_speed_sound0):
        return oth_relmach0 * stc_speed_sound0 - kin_W0
