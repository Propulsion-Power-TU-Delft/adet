"""
Equations that represent intermediate states computed
using control volume conservation equations, these
are not related to a node but associated to special
variables of either the inlet or outlet node
"""

import CoolProp as cp
import numpy as np

from adet.equations.base_equation import EquationBase, EquationConfig
from adet.equations.utils import minmax_bound, safe_min
from adet.variables import NodeVariables, ThermoVariables
from adet.varspec import VarSpec

n0 = NodeVariables(0)
n1 = NodeVariables(1)


_thrm = ThermoVariables()
ThroatVelocity = VarSpec('W_throat', 'm / s', node=0, guess=10)


class OptimalIncidence(EquationBase):
    config = EquationConfig(
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(n0.stc.Density.Glob,),
    )

    def residual(
        self,
        rho0: n0.stc.Density.Hint,
        Wm0: n0.kin.W_mer.Hint,
        bld_thick: n0.geo.BldThick.Hint,
        pitch: n0.geo.Pitch.Hint,
        met_angle0: n0.geo.MetalAngle.Hint,
        s0: n0.stc.Entropy.Hint,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        W_th0: ThroatVelocity.Hint,
        hh0: n0.geo.HDistr.Hint,
        beta_opt0: n0.kin.BetaOpt.Hint,
    ):
        hmass_th = h_rlt0 - W_th0**2 / 2
        Wm_th = W_th0 * np.cos(met_angle0)
        Wt_th = W_th0 * np.sin(met_angle0)

        # Isentropic throat density
        rho_th = self.eos(hmass_th, s0)

        original_area = hh0 * pitch
        restrict_area = hh0 * (pitch - bld_thick / np.cos(met_angle0))

        # U = const (same radius) => No Wt change = no Vt change
        r1 = rho0 * Wm0 * original_area - rho_th * Wm_th * restrict_area
        r2 = beta_opt0 - np.atan2(Wt_th, Wm0)

        return r1, r2


# NOTE: This is an experimental equation to compute the choking
# conditions in parallel to any row, it does not enforce anything
# for now but it is an accurate physical choking prediction that
# does not add overhead. In the future we could do something for
# massflow maximization using Lagrange multipliers like turboflow
class ChokingCriterion(EquationBase):
    manual_units = ('kg / s', 'm / s', 'Pa')
    input_pair = cp.HmassSmass_INPUTS
    output_quantities = ('rhomass', 'speed_sound', 'p')

    def residual(
        self,
        tot_hmass0,
        stc_smass0,
        area0,
        area_th,
        geo_metal_angle1,
        kin_U0,
        beta0,
        kin_U1,
        # Outputs
        W_thr,
        oth_p_choke1,
        W0,
    ):
        Wt_in = W_thr * np.sin(beta0)
        Wt_th = W0 * np.sin(geo_metal_angle1)

        Vt_in = Wt_in + kin_U0
        Vt_th = Wt_th + kin_U1

        Vm_in = W_thr * np.cos(beta0)
        Vm_th = W0 * np.cos(geo_metal_angle1)

        tot_hmass_th = tot_hmass0 + (kin_U1 * Vt_th - kin_U0 * Vt_in)
        stc_smass_th = stc_smass0

        W_thr = minmax_bound(W_thr, 0.1, 300)
        W0 = minmax_bound(W0, 0.1, 300)

        stc_hmass_in = tot_hmass0 - W_thr**2 / 2
        stc_hmass_th = tot_hmass_th - W0**2 / 2

        stc_rhomass_in, _, _ = self.eos(stc_hmass_in, stc_smass0)
        stc_rhomass_th, stc_speed_sound_th, stc_p_th = self.eos(
            stc_hmass_th, stc_smass_th
        )

        r1 = stc_rhomass_in * Vm_in * area0 - stc_rhomass_th * Vm_th * area_th

        # Assume velocity perpendicular to blade
        r2 = W0 - stc_speed_sound_th

        r3 = oth_p_choke1 - stc_p_th

        return r1, r2, r3


class ObliqueShock(EquationBase):
    def residual(
        self,
        W0: n0.kin.W_mag.Hint,
        W1: n1.kin.W_mag.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        rho0: n0.stc.Density.Hint,
        rho1: n1.stc.Density.Hint,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
        h0: n0.stc.Enthalpy.Hint,
        h1: n1.stc.Enthalpy.Hint,
        p0: n0.stc.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        mach0: n0.kin.Mach.Hint,
        ds_shock: n1.loss.Ds_shock.Hint,
        shock_angle: n1.oth.ShockAngle.Hint,
        defl_angle: n1.oth.ShockDeflection.Hint,
        gPv: n0.oth.GammaPV.Hint,
    ):
        w0 = W0 * np.cos(shock_angle)
        u0 = W0 * np.sin(shock_angle)

        w1 = W1 * np.cos(shock_angle - defl_angle)
        u1 = W1 * np.sin(shock_angle - defl_angle)

        # Continuity
        r1 = (rho1 * u1) - (rho0 * u0)
        # Tangential momentum
        r2 = (rho1 * u1 * w1) - (rho0 * u0 * w0)
        # Normal momentum
        r3 = (p1 + rho1 * u1**2) - (p0 + rho0 * u0**2)
        # Energy
        r4 = (h1 + u1**2 / 2) - (h0 + u0**2 / 2)
        # Link flow angles (no effect on shock)
        r5 = beta1 - (beta0 + defl_angle)

        # Flow angle expression - FIX the shock angle based on mach
        one_by_mach = safe_min(1 / mach0, 0.99)
        _r6 = shock_angle - (
            np.arcsin(one_by_mach)  # ty:ignore
            + (gPv + 1) / 4 * mach0**2 / (mach0**2 - 1) * defl_angle
        )

        _r7 = ds_shock - (s1 - s0)

        return (
            r1,
            r2,
            r3,
            r4,
            r5,
            # _r6,
            _r7,
        )


class LeadingEdgeThroat(EquationBase):
    config = EquationConfig(
        input_pair=cp.PT_INPUTS,
        out_properties=(
            _thrm.SpeedSound,
            _thrm.Density,
            _thrm.Entropy,
            _thrm.Enthalpy,
        ),
    )

    def residual(
        self,
        mf0: n0.oth.TotMassFlow.Hint,
        htr0: n0.rlt.Enthalpy.Hint,
        U0: n0.kin.BladeSpeed.Hint,
        s0: n0.stc.Entropy.Hint,
        rr0: n0.geo.RDistr.Hint,
        rr_hub: n0.geo.Rhub.Hint,
        rr_tip: n0.geo.Rtip.Hint,
        hh0: n0.geo.HDistr.Hint,
        # *** Throat quantities
        A_th: n0.geo.ThroatArea.Hint,
        # r_th: n0.geo.ThroatRadius.Hint,
        p_th: n0.oth.ThrPressure.Hint,
        mf_th: n0.oth.ThrMassFlow.Hint,
        mach_th: n0.kin.MachThroat.Hint,
        T_th: n0.oth.ThrTemperature.Hint,
        omega: n0.kin.Omega.Hint,
        # *** Geometry
        n_blades: n0.geo.NumBlades.Hint,
        met_angle: n0.geo.MetalAngle.Hint,
        bld_thick: n0.geo.BldThick.Hint,
    ):

        # TODO: Make some throat geometry spec
        rr_th = ((rr_hub**2 + rr_tip**2) / 2) ** 0.5
        U_th = omega * rr_th
        # ---

        a_th, rho_th, s_th, h_th = self.eos(p_th, T_th)
        w_th = mach_th * a_th
        r0 = mf_th - rho_th * w_th * A_th  # Throat massflow

        roth0 = htr0 - U0**2 / 2
        roth_th = h_th + w_th**2 / 2 - U_th**2 / 2

        # Main residuals
        r1 = mf0 - mf_th
        r2 = s0 - s_th
        r3 = roth0 - roth_th

        return r0, r1, r2, r3


class SimpleThroat(EquationBase):
    config = EquationConfig(
        input_pair=cp.PT_INPUTS,
        out_properties=(
            _thrm.SpeedSound,
            _thrm.Density,
            _thrm.Entropy,
            _thrm.Enthalpy,
        ),
    )

    def residual(
        self,
        mf0: n0.oth.StreamMassFlow.Hint,
        htr0: n0.rlt.Enthalpy.Hint,
        U0: n0.kin.BladeSpeed.Hint,
        omega: n0.kin.Omega.Hint,
        s0: n0.stc.Entropy.Hint,
        # *** Throat quantities
        A_th: n0.geo.ThroatArea.Hint,
        p_th: n0.oth.ThrPressure.Hint,
        rr_th: n0.geo.ThroatRadius.Hint,
        mf_th: n0.oth.ThrMassFlow.Hint,
        mach_th: n0.kin.MachThroat.Hint,
        T_th: n0.oth.ThrTemperature.Hint,
    ):

        U_th = omega * rr_th

        a_th, rho_th, s_th, h_th = self.eos(p_th, T_th)
        w_th = mach_th * a_th
        r0 = mf_th - rho_th * w_th * A_th  # Throat massflow

        roth0 = htr0 - U0**2 / 2
        roth_th = h_th + w_th**2 / 2 - U_th**2 / 2

        # Main residuals
        r1 = mf0 - mf_th
        r2 = s0 - s_th
        r3 = roth0 - roth_th

        return r0, r1, r2, r3


class ChokingArea(EquationBase):
    config = EquationConfig(
        input_pair=cp.PT_INPUTS,
        out_properties=(
            _thrm.SpeedSound,
            _thrm.Density,
            _thrm.Entropy,
            _thrm.Enthalpy,
        ),
    )

    def residual(
        self,
        mf0: n0.oth.TotMassFlow.Hint,
        htr0: n0.rlt.Enthalpy.Hint,
        U0: n0.kin.BladeSpeed.Hint,
        omega: n0.kin.Omega.Hint,
        s0: n0.stc.Entropy.Hint,
        rr_hub: n0.geo.Rhub.Hint,
        rr_tip: n0.geo.Rtip.Hint,
        # *** Throat quantities
        A_chk: n0.geo.ChokeArea.Hint,
        p_chk: n0.oth.ChkPressure.Hint,
        T_chk: n0.oth.ChkTemperature.Hint,
    ):

        rr_chk = ((rr_hub**2 + rr_tip**2) / 2) ** 0.5
        U_chk = omega * rr_chk
        # ---

        a_chk, rho_chk, s_chk, h_chk = self.eos(p_chk, T_chk)
        w_chk = a_chk

        roth0 = htr0 - U0**2 / 2
        roth_chk = h_chk + w_chk**2 / 2 - U_chk**2 / 2

        # Main residuals
        r1 = mf0 - rho_chk * w_chk * A_chk
        r2 = s0 - s_chk
        r3 = roth0 - roth_chk

        return r1, r2, r3
