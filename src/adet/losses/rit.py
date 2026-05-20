from adet.equations.utils import safe_max
from adet.equations.base_equation import EquationConfig
from adet.variables import NodeVariables, ThermoVariables
import CoolProp as cp
from adet.losses.base_loss import LossModel
import numpy as np


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
        roth_minus_his = safe_max(Roth0 - h1_is, 0.001 * h1_is)
        W1_is = (2 * roth_minus_his + U1**2) ** 0.5
        W1_lss = W1_is * (1 - zeta_2D) ** 0.5
        h1_lss = Roth0 - W1_lss**2 / 2 + U1**2 / 2
        s1_profile = self.eos(h1_lss, p1)

        return Ds_prof - (s1_profile - s0)


class StatorEndwallLoss(LossModel):
    def residual(self): ...


class ShockLoss(LossModel):
    def residual(
        self,
        cp: n1.stc.Cp.Hint,
        s0: n0.stc.Entropy.Hint,
        M1: n1.kin.RelMach.Hint,
        p1: n1.stc.Pressure.Hint,
        gPv1: n1.oth.GammaPV.Hint,
        mmass: n1.stc.MolarMass.Hint,
        R_un: n1.stc.GasConstant.Hint,
        shock_angle: n1.oth.ShockAngle.Hint,
    ):
        R = R_un / mmass

        theta = np.arctan(
            2
            / np.tan(shock_angle)
            * (M1**2 * np.sin(shock_angle) ** 2 - 1)
            / (M1**2 * (gPv1 + np.cos(2 * shock_angle) + 2))
        )
        if M1 * np.sin(shock_angle) >= 1.0:
            M1_normal = M1 * np.sin(shock_angle)
            M2_normal = np.sqrt(
                (1 + (gPv1 - 1) / 2 * M1_normal**2)
                / (gPv1 * M1_normal**2 - (gPv1 - 1) / 2)
            )
            M2 = M2_normal / np.sin(shock_angle)
            P_ratio = 1 + (2 * gPv1) / (gPv1 + 1) * (M1_normal**2 - 1)
            D_ratio = (1 + (gPv1 + 1) / (gPv1 - 1) * P_ratio) / (
                (gPv1 + 1) / (gPv1 - 1) + P_ratio
            )
            T_ratio = P_ratio / D_ratio
            delta_s = cp * np.log(T_ratio) - R * np.log(P_ratio)
        else:
            M2 = M1
            P_ratio = 1
            T_ratio = 1
            delta_s = 0

            # Shock angle computation
            shock_angle_new = (
                np.arcsin(1 / M1) + (gPv1 + 1) / 4 * M1**2 / (M1**2 - 1) * theta
            )
            dshock_angle = (shock_angle - shock_angle_new) / shock_angle
            shock_angle = shock_angle_new
            theta = 0


# *** Impeller
class ImpellerPassageLoss(LossModel):
    def residual(self): ...


class ImpellerLeakageLoss(LossModel):
    def residual(self): ...


class GlassmanMeitnerBlayer(LossModel):
    def residual(self): ...


class RadialGapLoss(LossModel):
    def residual(self): ...
