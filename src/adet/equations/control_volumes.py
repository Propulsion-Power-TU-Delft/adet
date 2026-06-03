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


class FullIncidence(EquationBase):
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
        geo_eff_area0,
        geo_eff_area1,
        geo_metal_angle1,
        kin_U0,
        geo_metal_angle0,
        kin_U1,
        # Outputs
        kin_W_choke0,
        kin_W_choke1,
        oth_p_choke1,
    ):
        Wt_in = kin_W_choke0 * np.sin(geo_metal_angle0)
        Wt_th = kin_W_choke1 * np.sin(geo_metal_angle1)

        Vt_in = Wt_in + kin_U0
        Vt_th = Wt_th + kin_U1

        Vm_in = kin_W_choke0 * np.cos(geo_metal_angle0)
        Vm_th = kin_W_choke1 * np.cos(geo_metal_angle1)

        tot_hmass_th = tot_hmass0 + (kin_U1 * Vt_th - kin_U0 * Vt_in)
        stc_smass_th = stc_smass0

        kin_W_choke0 = minmax_bound(kin_W_choke0, 0.1, 300)
        kin_W_choke1 = minmax_bound(kin_W_choke1, 0.1, 300)

        stc_hmass_in = tot_hmass0 - kin_W_choke0**2 / 2
        stc_hmass_th = tot_hmass_th - kin_W_choke1**2 / 2

        stc_rhomass_in, _, _ = self.eos(stc_hmass_in, stc_smass0)
        stc_rhomass_th, stc_speed_sound_th, stc_p_th = self.eos(
            stc_hmass_th, stc_smass_th
        )

        r1 = (
            stc_rhomass_in * Vm_in * geo_eff_area0
            - stc_rhomass_th * Vm_th * geo_eff_area1
        )

        # Assume velocity perpendicular to blade
        r2 = kin_W_choke1 - stc_speed_sound_th

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

        # Flow angle expression - FIX the shock angle
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


LagMult1 = VarSpec('lamb1', '', node=0, guess=0.0)
LagMult2 = VarSpec('lamb2', '(kg / s) * (J / kg / K)**-1', node=0, guess=0.0)
LagMult3 = VarSpec('lamb3', '(kg / s) * (J / kg)**-1', node=0, guess=0.0)


class ThroatConditions(EquationBase):
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
        mf0: n0.oth.MassFlow.Hint,
        htr0: n0.rlt.Enthalpy.Hint,
        U0: n0.kin.BladeSpeed.Hint,
        s0: n0.stc.Entropy.Hint,
        # Throat quantities
        A_th: n0.geo.ThroatArea.Hint,
        p_th: n0.oth.ThrPressure.Hint,
        mf_th: n0.oth.ThrMassFlow.Hint,
        mach_th: n0.kin.MachThroat.Hint,
        T_th: n0.oth.ThrTemperature.Hint,
        omega: n0.kin.Omega.Hint,
        r_th: n0.geo.ThroatRadius.Hint,
        # lamb1: LagMult1.Hint,
        # lamb2: LagMult2.Hint,
        # lamb3: LagMult3.Hint,
    ):

        a_th, rho_th, s_th, h_th = self.eos(p_th, T_th)
        w_th = mach_th * a_th
        r0 = mf_th - rho_th * w_th * A_th  # Massflow definition

        roth0 = htr0 - U0**2 / 2
        roth_th = h_th + w_th**2 / 2 - U0**2 / 2

        # Main residuals
        r1 = mf0 - mf_th
        r2 = s0 - s_th
        r3 = roth0 - roth_th

        # Lagrangian step
        # lagr = mf_th + lamb1 * r1 + lamb2 * r2 + lamb3 * r3
        # variables = [Wm0, mach_th, p_th]
        # r_lagr = safe_gradient(lagr, variables)
        #
        # residuals.extend(r_lagr)

        return r0, r1, r2, r3
