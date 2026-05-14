from adet.equations.utils import thermo_deriv
from adet.equations.base_equation import EquationConfig
from adet.equations.variables import NodeVariables, ThermoVariables
import CoolProp as cp
from adet.losses.base_loss import LossModel
import numpy as np


n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


def compute_pr_her_params(Q): ...


# *** Stator
class StatorProfileLoss(LossModel):
    config = EquationConfig(
        manual_units=(),
        input_pair=cp.DmassSmass_INPUTS,
        out_properties=(thrm.Pressure,),
    )

    def residual(
        self,
        angle_out,
        bld_thick: n1.geo.BldThick.Hint,
        mom_th: n1.oth.MomThick.Hint,
        pitch: n1.geo.Pitch.Hint,
        rho0: n0.stc.Density.Hint,
        s0: n0.stc.Entropy.Hint,
        p0: n0.stc.Pressure.Hint,
        rho1: n1.stc.Density.Hint,
        s1: n1.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        V_out: n1.kin.W_mag.Hint,
        Tt_in: n0.tot.Temperature.Hint,
        gas_const: n1.stc.GasConstant.Hint,
        mmass: n1.stc.MolarMass.Hint,
    ):

        # GammaPV computations
        dp_drho0 = thermo_deriv(self.eos, rho0, s0, 0)[0]
        gPv_in = rho0 / p0 * dp_drho0

        dp_drho1 = thermo_deriv(self.eos, rho1, s1, 1)[0]
        gPv_out = rho1 / p1 * dp_drho1

        R = gas_const / mmass
        gPv = (gPv_in + gPv_out) / 2
        V_cr = (2 * gPv / (gPv + 1) * R * Tt_in) ** 0.5

        Q = (gPv - 1) / (gPv + 1) * (V_out / V_cr) ** 2

        E, H = compute_pr_her_params(Q)  # Prust-Hermann parameters
        t_s = bld_thick / pitch
        theta_s = mom_th / pitch

        zeta_2D = (E * theta_s) / (np.cos(angle_out) - t_s - H * theta_s)


class StatorEndwallLoss(LossModel):
    def residual(self): ...


class ShockLoss(LossModel):
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
