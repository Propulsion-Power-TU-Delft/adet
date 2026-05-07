"""
Module that gathers equations that represent definitions of nondimensional
coefficients used in TurboMachinery
"""

import numpy as np
import CoolProp as cp

from adet.equations import EquationBase
from adet.equations.base_equation import EquationConfig
from adet.equations.utils import get_midspan_idx, safe_abs, thermo_deriv
from adet.equations.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)


class TotalTotalPressureRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def residual(
        self,
        p0_tot: n0.tot.Pressure.Hint,
        p1_tot: n1.tot.Pressure.Hint,
        p_ratio_tt1: n1.ndim.PRatioTT.Hint,
    ):
        return p0_tot * p_ratio_tt1 - p1_tot


class StaticTotalPressRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def residual(
        self,
        p0_tot: n0.tot.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        p_ratio_ts1: n1.ndim.PRatioTS.Hint,
    ):
        return p0_tot * p_ratio_ts1 - p1


class VolumetricFlowRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def residual(
        self,
        rho0_tot: n0.tot.Density.Hint,
        rho1: n1.stc.Density.Hint,
        vol_ratio1: n1.ndim.VolflowRatio.Hint,
    ):
        midspan = get_midspan_idx(rho1)
        vol_ratio = rho0_tot[midspan] / rho1[midspan]
        return vol_ratio1 - vol_ratio


class TotalTotalExpansionEfficiency(EquationBase):
    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(ThermoVariables().Enthalpy,),
        manual_units=('dimensionless',),
    )

    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        p1_tot: n1.tot.Pressure.Hint,
        h0_tot: n0.tot.Enthalpy.Hint,
        h1_tot: n1.tot.Enthalpy.Hint,
        eta_tt1: n1.ndim.EtaTT.Hint,
    ):
        h_is1 = self.eos(p1_tot, s0)
        return eta_tt1 - (h0_tot - h1_tot) / (h0_tot - h_is1)


class TotalTotalCompressionEfficiency(EquationBase):
    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(ThermoVariables().Enthalpy,),
        manual_units=('J / kg',),
    )

    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        p1_tot: n1.tot.Pressure.Hint,
        h0_tot: n0.tot.Enthalpy.Hint,
        h1_tot: n1.tot.Enthalpy.Hint,
        eta_tt1: n1.ndim.EtaTT.Hint,
    ):
        h_is1 = self.eos(p1_tot, s0)
        return eta_tt1 * (h1_tot - h0_tot) - (h_is1 - h0_tot)


class TotalStaticLoadingCoefficient(EquationBase):
    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(ThermoVariables().Enthalpy,),
        manual_units=('J / kg',),
    )

    def residual(
        self,
        ts_load1: n1.ndim.TSLoadCoeff.Hint,
        p1: n1.stc.Pressure.Hint,
        s0: n0.stc.Entropy.Hint,
        h0_tot: n0.tot.Enthalpy.Hint,
        u1: n1.kin.BladeSpeed.Hint,
    ):
        midspan = get_midspan_idx(p1)
        h_is1 = self.eos(p1, s0)[midspan]
        return u1[midspan] ** 2 * ts_load1 - 2 * (h0_tot[midspan] - h_is1)


class StaticPressRatio(EquationBase):
    """
    .. math::
        \\beta_{tt} = \\frac{p_{t1}}{p_{t,0}}
    """

    def residual(
        self,
        p0: n0.stc.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        p_ratio1: n1.ndim.PRatio.Hint,
    ):
        return p0 * p_ratio1 - p1


class StaticTotalDegreeOfReaction(EquationBase):
    """
    0 - [Stator] - 1 === 2 - [Rotor] - 3
    This assumes the stator is on nodes 0,1 and the stator on 2,3 is the rotor.
    The degree of reaction is an `oth` property of node 3
    """

    def residual(
        self,
        h0_tot: n0.tot.Enthalpy.Hint,
        h1: n1.stc.Enthalpy.Hint,
        h2: n2.stc.Enthalpy.Hint,
        h3: n3.stc.Enthalpy.Hint,
        h3_tot: n3.tot.Enthalpy.Hint,
        react_ts3: n3.ndim.DegreeOfReactionTS.Hint,
    ):
        midspan = get_midspan_idx(h0_tot)

        delta_h_rotor = h3 - h2
        delta_h_tot_stage = h3_tot - h0_tot

        return (
            delta_h_tot_stage[midspan] * react_ts3
            - delta_h_rotor[midspan]
        )


class StaticDegreeOfReaction(EquationBase):
    """
    0 - [Stator] - 1 === 2 - [Rotor] - 3
    This assumes the stator is on nodes 0,1 and the stator on 2,3 is the rotor.
    The degree of reaction is an `oth` property of node 3
    """

    def residual(
        self,
        h0: n0.stc.Enthalpy.Hint,
        h1: n1.stc.Enthalpy.Hint,
        h2: n2.stc.Enthalpy.Hint,
        h3: n3.stc.Enthalpy.Hint,
        react3: n3.ndim.DegreeOfReaction.Hint,
    ):
        delta_h_rotor = h3 - h2
        delta_h_stage = h3 - h0

        return delta_h_stage * react3 - delta_h_rotor


class DensityRatio(EquationBase):
    """
    .. math::
        \\mathrm{FR} = \\frac{\\rho_{t1}}{p_{t,0}}
    """

    def residual(
        self,
        p0_tot: n0.tot.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        rho_ratio1: n1.ndim.RhoRatio.Hint,
    ):
        return rho_ratio1 - p1 / p0_tot


class FlowCoefficient(EquationBase):
    """
    .. math::
        \\phi = \\frac{V_{m0}}{U_{0}}
    """

    def residual(
        self,
        vm0: n0.kin.V_mer.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        flow_coeff1: n1.ndim.FlowCoeff.Hint,
    ):
        midspan = get_midspan_idx(vm0)

        return safe_abs(u1[midspan]) * flow_coeff1 - vm0[midspan]


class WorkCoefficient(EquationBase):
    """
    .. math::
        \\psi = \\frac{\\Delta h_t}{U_1^2}

    Note
    ----
    In some literature the denominator is :math:`2U_0V_{t0}`
    """

    def residual(
        self,
        h0_tot: n0.tot.Enthalpy.Hint,
        h1_tot: n1.tot.Enthalpy.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        work_coeff1: n1.ndim.WorkCoeff.Hint,
    ):
        midspan = get_midspan_idx(h0_tot)

        return u1[midspan] ** 2 * work_coeff1 - (
            h1_tot[midspan] - h0_tot[midspan]
        )


class EnthalpyDropCoefficient(EquationBase):
    def residual(
        self,
        h0: n0.stc.Enthalpy.Hint,
        h1: n1.stc.Enthalpy.Hint,
        v1: n1.kin.V_mag.Hint,
        hdrop_coeff1: n1.ndim.HdropCoeff.Hint,
    ):
        return v1**2 * hdrop_coeff1 - (h1 - h0)


class SwallowingCapacity(EquationBase):
    """
    .. math::
        \\phi_{t0} = \\frac{\\dot{m}}{\\rho_{t0} D_1^2 U_1}


    pg 254 Casey - Radial Flow Turbocompressors
    """

    def residual(
        self,
        u1: n1.kin.BladeSpeed.Hint,
        rr1: n1.geo.RDistr.Hint,
        rho0_tot: n0.tot.Density.Hint,
        swll_cap0: n0.ndim.SwallowingCap.Hint,
        mf0: n0.oth.MassFlow.Hint,
    ):
        return swll_cap0 - mf0 / (
            rho0_tot * u1 * (2 * rr1) ** 2
        )


class SpecificSpeed(EquationBase):
    def residual(
        self,
        spec_speed1: n1.ndim.SpecificSpeed.Hint,
        omega1: n1.kin.Omega.Hint,
        mf1: n1.oth.MassFlow.Hint,
        rho1: n1.stc.Density.Hint,
        h0_tot: n0.tot.Enthalpy.Hint,
        h1: n1.stc.Enthalpy.Hint,
    ):
        return spec_speed1 * (
            (h0_tot - h1) ** (3 / 4)
        ) - omega1 * np.sqrt(mf1 / rho1)


class SizeParameter(EquationBase):
    def residual(
        self,
        size_param1: n1.ndim.SizeParameter.Hint,
        mf1: n1.oth.MassFlow.Hint,
        rho1: n1.stc.Density.Hint,
        h0_tot: n0.tot.Enthalpy.Hint,
        h1: n1.stc.Enthalpy.Hint,
    ):
        return (
            size_param1 * ((h0_tot - h1) ** (1 / 4))
            - (mf1 / rho1) ** 0.5
        )


class AbsoluteMachNumber(EquationBase):
    def residual(
        self,
        mach0: n0.kin.Mach.Hint,
        mer_mach0: n0.kin.MerMach.Hint,
        vm0: n0.kin.V_mer.Hint,
        v0: n0.kin.V_mag.Hint,
        a_sound0: n0.stc.SpeedSound.Hint,
    ):
        r1 = mach0 - v0 / a_sound0
        r2 = mer_mach0 - vm0 / a_sound0
        return r1, r2


class RelativeMachNumber(EquationBase):
    config = EquationConfig(manual_units=('dimensionless',))

    def residual(
        self,
        rel_mach0: n0.kin.RelMach.Hint,
        w0: n0.kin.W_mag.Hint,
        a_sound0: n0.stc.SpeedSound.Hint,
    ):
        return rel_mach0 * a_sound0 - w0


class GammaPV(EquationBase):
    # WARN: This update pair for ideal gas
    # Not that using gamma_pv makes sense, but
    # beware
    config = EquationConfig(
        input_pair=cp.DmassSmass_INPUTS,
        out_properties=(ThermoVariables().Pressure,),
        manual_units=('dimensionless',),
    )

    def residual(
        self,
        gamma_pv0: n0.oth.GammaPV.Hint,
        rho0: n0.stc.Density.Hint,
        s0: n0.stc.Entropy.Hint,
        p0: n0.stc.Pressure.Hint,
    ):
        dp_drho = thermo_deriv(self.eos, rho0, s0, 0)[0]
        return gamma_pv0 - rho0 / p0 * dp_drho
