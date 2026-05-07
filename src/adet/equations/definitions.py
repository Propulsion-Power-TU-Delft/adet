"""
Simple quantity defintions, for defining differences or ratios rather than
the single quantities
"""

import CoolProp as cp
import numpy as np

from adet.equations.base_equation import EquationBase, EquationConfig
from adet.equations.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)
thrm = ThermoVariables()


class AngleDeflection(EquationBase):
    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        defl1: n1.kin.Deflection.Hint,
    ):
        return defl1 - (beta1 - beta0)


class IncidenceAngle(EquationBase):
    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        inc_angle0: n0.kin.IncAngle.Hint,
        metal_angle0: n0.geo.MetalAngle.Hint,
    ):
        return inc_angle0 - (beta0 - metal_angle0)


class DeviationAngle(EquationBase):
    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        dev_angle1: n1.kin.DevAngle.Hint,
        metal_angle0: n0.geo.MetalAngle.Hint,
    ):
        return dev_angle1 + np.sign(metal_angle0) * (beta1 - beta0)


class OptimalIncidence(EquationBase):
    """
    Angle for which there is no change in tangential velocity due to
    blade blockage
    """

    def residual(
        self,
        wt1: n1.kin.W_tan.Hint,
        wm0: n0.kin.W_mer.Hint,
        beta_opt0: n0.kin.BetaOpt.Hint,
    ):

        return beta_opt0 - np.atan2(wt1, wm0)


class RepeatedStage(EquationBase):
    """0 - [Stator] - 1 = 2 - [Rotor] - 3"""

    def residual(
        self,
        alpha0: n0.kin.FlowAngleAbs.Hint,
        alpha3: n3.kin.FlowAngleAbs.Hint,
        vm0: n0.kin.V_mer.Hint,
        vm1: n1.kin.V_mer.Hint,
        vm2: n2.kin.V_mer.Hint,
        vm3: n3.kin.V_mer.Hint,
    ):
        r1 = alpha0 - alpha3
        r2 = vm3 - vm2
        r3 = vm1 - vm0

        return r1, r2, r3


class MeridionalVelocityRatio(EquationBase):
    def residual(
        self,
        vm0: n0.kin.V_mer.Hint,
        vm1: n1.kin.V_mer.Hint,
        vm_ratio1: n1.kin.VmRatio.Hint,
    ):
        return vm0 * vm_ratio1 - vm1


class MidspanVelocities(EquationBase):
    def residual(
        self,
        kin_V0,
        kin_Vm0,
        kin_Vt0,
        kin_V_midspan0,
        kin_Vm_midspan0,
        kin_Vt_midspan0,
    ):
        num_span = max(kin_V0.shape)
        if num_span == 1:
            midspan = 0
        else:
            midspan = num_span // 2

        r1 = kin_V_midspan0 - kin_V0[midspan]
        r2 = kin_Vm_midspan0 - kin_Vm0[midspan]
        r3 = kin_Vt_midspan0 - kin_Vt0[midspan]

        return r1, r2, r3


class EffectiveBladeNumber(EquationBase):
    def residual(
        self,
        n_blades0: n0.geo.NumBlades.Hint,
        n_splitters0: n0.geo.NumSplitters.Hint,
        n_blades_eff0: n0.geo.NumBladesEff.Hint,
    ):
        return n_blades_eff0 - (n_blades0 + 0.75 * n_splitters0)


class IsentropicProperties(EquationBase):
    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(thrm.Enthalpy, thrm.Temperature),
        manual_units=('J / kg', 'K', 'J / kg', 'K'),
    )

    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        p1_tot: n1.tot.Pressure.Hint,
        p1_stc: n1.stc.Pressure.Hint,
        h_is_tot1: n1.oth.Enthalpy_totIs.Hint,
        T_is_tot1: n1.oth.Tis_tot.Hint,
        h_is1: n1.oth.Enthalpy_Is.Hint,
        T_is1: n1.oth.Tis_stc.Hint,
    ):
        h_tot_is, T_tot_is = self.eos(p1_tot, s0)
        h_stc_is, T_stc_is = self.eos(p1_stc, s0)

        r1 = h_is_tot1 - h_tot_is
        r2 = T_is_tot1 - T_tot_is
        r3 = h_is1 - h_stc_is
        r4 = T_is1 - T_stc_is

        return r1, r2, r3, r4


class ClearanceByHeight(EquationBase):
    def residual(
        self,
        clr_by_h0: n0.geo.ClearanceByHeight.Hint,
        h0: n0.geo.Height.Hint,
        tip_clr0: n0.geo.TipClearance.Hint,
    ):
        return clr_by_h0 * h0 - tip_clr0


class ReducedThermoQuantities(EquationBase):
    def residual(
        self,
        T_tot0: n0.tot.Temperature.Hint,
        p_tot0: n0.tot.Pressure.Hint,
        T_red_tot0: n0.oth.TotTRed.Hint,
        p_red_tot0: n0.oth.TotPRed.Hint,
        p_crit0: n0.stc.CriticalTemp.Hint,
        T_crit0: n0.stc.CriticalTemperature.Hint,
    ):
        r1 = p_tot0 - p_red_tot0 * p_crit0
        r2 = T_tot0 - T_red_tot0 * T_crit0

        return r1, r2


class BoundaryLayerRatios(EquationBase):
    """Boundary layer properties ratios definitions
    based on trailing edge thickness"""

    def residual(
        self,
        h0: n0.geo.Height.Hint,
        bld_thick0: n0.geo.BldThick.Hint,
        mom_thick0: n0.oth.MomThick.Hint,
        disp_thick0: n0.oth.DispThick.Hint,
        disp_thick_ew0: n0.oth.DispThickEW.Hint,
        disp_by_mom0: n0.oth.DispByMom.Hint,
        mom_by_bld0: n0.oth.MomByBld.Hint,
        disp_by_hgt0: n0.oth.DispByHgt.Hint,
    ):
        r1 = disp_thick0 - disp_by_mom0 * mom_thick0
        r2 = mom_thick0 - mom_by_bld0 * bld_thick0
        r3 = disp_thick_ew0 - disp_by_hgt0 * h0

        return r1, r2, r3
