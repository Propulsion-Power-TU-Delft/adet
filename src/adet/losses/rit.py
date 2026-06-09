import CoolProp as cp
import numpy as np

from adet.equations.base_equation import DeviationModel, EquationConfig
from adet.equations.utils import safe_abs, safe_max
from adet.losses.base_loss import LossModel
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


def computePrustHerman(Q):

    den = 1 / 1.68 + Q / 2.88 + Q**2 / 4.4 + Q**3 / 6.24
    # Energy factor
    numE = 2 * (1 / 1.92 + Q / 3.2 + Q**2 / 4.8 + Q**3 / 6.72)
    E = numE / den
    # Form factor
    numH = 1 / 1.2 + 3 * Q / 1.6 + 5 * Q**2 / 2 + 7 * Q**3 / 2.4 + 9 * Q**4 / 2.8
    H = numH / den

    return E, H


# *** Stator
class StatorProfileLoss(LossModel):
    """
    From `COMPUTER PROGRAM FOR DESIGN ANALYSIS OF RADIAL-INFLOW TURBINES`
    Arthur J. Glassman
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )
    # on both nodes. For ideal gas need to use

    def residual(
        self,
        angle_out: n1.kin.FlowAngleRel.Hint,
        bld_thick: n1.geo.BldThick.Hint,
        mom_th: n1.oth.MomThick.Hint,
        pitch: n1.geo.Pitch.Hint,
        s0: n0.stc.Entropy.Hint,
        h0: n0.stc.Enthalpy.Hint,
        W1: n1.kin.W_mag.Hint,
        Tt0: n0.tot.Temperature.Hint,
        gas_const: n1.stc.GasConstant.Hint,
        mmass: n1.stc.MolarMass.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        W0: n0.kin.W_mag.Hint,
        U0: n0.kin.BladeSpeed.Hint,
        U1: n1.kin.BladeSpeed.Hint,
        p1: n1.stc.Pressure.Hint,
        gPv0: n0.oth.GammaPV.Hint,
        gPv1: n1.oth.GammaPV.Hint,
        Ds_prof: n1.loss.Ds_profile.Hint,
    ):

        R = gas_const / mmass
        gPv = (gPv0 + gPv1) / 2
        W_cr = (2 * gPv / (gPv + 1) * R * Tt0) ** 0.5

        # Loss coefficient computation
        Q = (gPv - 1) / (gPv + 1) * (W1 / W_cr) ** 2
        E, H = computePrustHerman(Q)
        t_s = bld_thick / pitch
        theta_s = mom_th / pitch
        zeta_2D = (E * theta_s) / (np.cos(angle_out) - t_s - H * theta_s)

        Roth0 = h0 + W0**2 / 2 - U0**2 / 2
        # Avoid negative square argument
        roth_minus_his = safe_max(Roth0 - h1_is, 0.0 * h1_is)
        W1_is = (2 * roth_minus_his + U1**2) ** 0.5
        W1_lss = W1_is * (1 - zeta_2D) ** 0.5
        h1_lss = Roth0 - W1_lss**2 / 2 + U1**2 / 2
        s1_profile = self.eos(h1_lss, p1)

        return Ds_prof - (s1_profile - s0)


class ShockLoss(DeviationModel):
    config = EquationConfig(
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(thrm.Pressure, thrm.Density, thrm.SpeedSound),
    )

    def residual(
        self,
        # *** Shock properties
        mach0: n0.kin.MachPresh.Hint,
        h0: n0.oth.EnthalpyPresh.Hint,
        s0: n0.oth.EntropyPresh.Hint,
        s1: n0.stc.Entropy.Hint,
        shock_angle: n0.oth.ShockAngle.Hint,
        defl_angle: n0.oth.ShockDeflection.Hint,
        metal_ang: n0.geo.MetalAngle.Hint,
        beta1: n0.kin.FlowAngleRel.Hint,
        # *** Outlet
        p1: n0.stc.Pressure.Hint,
        rho1: n0.stc.Density.Hint,
        h1: n0.stc.Enthalpy.Hint,
        W1: n0.kin.W_mag.Hint,
        gPv: n0.oth.GammaPV.Hint,
        ds_shock: n0.loss.Ds_shock.Hint,
    ):
        p0, rho0, a0 = self.eos(h0, s0)
        W0 = a0 * mach0

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

        # Shock angle (arcsin is defined only below 1)
        mach0 = safe_abs(W0 / a0)
        r5 = shock_angle - (
            np.arcsin(1 / mach0)
            + (gPv + 1) / 4 * mach0**2 / (mach0**2 - 1) * defl_angle
        )

        r6 = ds_shock - (s1 - s0)
        r7 = beta1 - (metal_ang - defl_angle)

        return r1, r2, r3, r4, r5, r6, r7


class StatorEndwallLoss(LossModel):
    def residual(self): ...


# *** Impeller
class ImpellerPassageLoss(LossModel):
    def residual(self): ...


class ImpellerLeakageLoss(LossModel):
    def residual(self): ...


class GlassmanMeitnerBlayer(LossModel):
    def residual(self): ...


class RadialGapLoss(LossModel):
    def residual(self): ...
