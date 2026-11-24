"""Deviation models: flow-angle deviation from the blade metal angle (slip and related models) for compressors and turbines."""

import CoolProp as cp
import numpy as np

from adet.equations.base_equation import (
    DeviationModel,
    EquationConfig,
)
from adet.equations.utils import (
    get_midspan_idx,
    safe_abs,
    safe_if_else,
)
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


class FreeDeviation(DeviationModel):
    """
    Decouple the flow angle and the metal angle,
    useful when the deviation is fully determined
    by the thermodynamic state
    """

    def residual(self, beta: n0.kin.FlowAngleRel.Hint):
        return ()


class ZeroDeviation(DeviationModel):
    """Impose equality between kinematic and geometric angles at a node"""

    def residual(
        self,
        met_angle0: n0.geo.MetalAngle.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
    ):
        return beta0 - met_angle0


class ZeroMidspanDeviation(DeviationModel):
    def residual(self, kin_beta0, geo_metal_angle0):
        midspan = get_midspan_idx(kin_beta0)
        return kin_beta0[midspan] - geo_metal_angle0[midspan]


class BackstromSlip(DeviationModel):
    def residual(
        self,
        slip1: n1.oth.SlipFactor.Hint,
        slip_coeff1: n1.oth.SlipFactCoeff.Hint,
        rr0: n0.geo.RDistr.Hint,
        rr1: n1.geo.RDistr.Hint,
        n_bl1: n1.geo.NumBlades.Hint,
        metal_ang1: n1.geo.MetalAngle.Hint,
        eff_solid1: n1.geo.EffSolidity.Hint,
        n_split1: n1.geo.NumSplitters.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        wm1: n1.kin.W_mer.Hint,
        wt1: n1.kin.W_tan.Hint,
    ):
        tot_blades = n_bl1 + n_split1

        radius_ratio = rr0 / rr1
        r1 = eff_solid1 - (
            (1 - radius_ratio) * tot_blades / (2 * np.pi * np.cos(metal_ang1))
        )

        r2 = slip1 - (
            1 - 1 / (1 + slip_coeff1 * eff_solid1 * np.cos(metal_ang1) ** 0.5)
        )

        slip_velocity = u1 * (1 - slip1)

        wt_noslip = wm1 * np.tan(metal_ang1)

        r3 = wt1 - (wt_noslip - slip_velocity)

        return r1, r2, r3


class QiuSlip(DeviationModel):
    """
    Qiu et al. (2011) - Analysis and Validation of a Unified Slip Factor Model
    for Impellers at Design and Off-Design Conditions.

    Valid for radial and mixed-flow impeller outlets: the meridional inclination
    angle at the exit (MeridionalAngle, 0 = axial, 90 deg = radial) enters the
    F-factor and the radial term. The blade turning and passage width variation
    terms, Eqs. (10b) and (10c), are neglected.

    Conventions as in the paper:
    - Metal angle from the meridional direction, negative if backswept
    - Exit pitch s2 = 2 pi r2 / Z2, evaluated at each spanwise station, with the
        same blade number Z2 used in the radial term (effective blade number)
    - Trailing edge thickness normal to the blade
    - shape: F-factor, Eq. (6), with the factor 2 of the derivation in the appendix
    - r1: slip-factor residual, Eq. (8), retaining only the radial term in Eq. (10a)
    - slip_velocity: American slip-factor definition
    - r2: exit velocity-triangle closure

    Single-node equation: all quantities belong to the impeller outlet, apply it
    at the outlet node of the blade row, e.g. `QiuSlip(): 1`.
    """

    def residual(
        self,
        slip0: n0.oth.SlipFactor.Hint,
        metal_ang0: n0.geo.MetalAngle.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
        n_bl_eff0: n0.geo.NumBladesEff.Hint,
        bld_thick0: n0.geo.BldThick.Hint,
        rr0: n0.geo.RDistr.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        wm0: n0.kin.W_mer.Hint,
        wt0: n0.kin.W_tan.Hint,
    ):
        sin_incl = np.sin(mer_angle0)
        exit_pitch = 2 * np.pi * rr0 / n_bl_eff0

        shape = (
            1
            - 2
            * np.sin(np.pi / n_bl_eff0)
            * np.sin(np.pi / n_bl_eff0 + metal_ang0)
            * np.cos(metal_ang0)
            * sin_incl
            - bld_thick0 / (exit_pitch * np.cos(metal_ang0))
        )

        r1 = slip0 - (1 - shape * np.pi * np.cos(metal_ang0) * sin_incl / n_bl_eff0)

        slip_velocity = u0 * (1 - slip0)

        wt_noslip = wm0 * np.tan(metal_ang0)

        r2 = wt0 - (wt_noslip - slip_velocity)

        return r1, r2


def aungier_delta0(met_angle):
    """Aungier incompressible deviation angle delta0 [rad] at the blade metal angle"""
    cos_beta = np.cos(met_angle)  # > 0
    beta_abs = safe_abs(met_angle)  # > 0
    return beta_abs - np.arccos(
        cos_beta * (1 + (1 - cos_beta) * (2 * beta_abs / np.pi) ** 2)
    )


def aungier_delta_sub(delta0, mach_out):
    """Aungier subsonic deviation delta_sub [rad], delta0 corrected for outlet Mach"""
    X = 2 * mach_out - 1
    return delta0 * (1 - 10 * X**3 + 15 * X**4 - 6 * X**5)


class AungierDeviationModel(DeviationModel):
    """Only valid for subsonic deviation"""

    def residual(
        self,
        met_angle: n0.geo.MetalAngle.Hint,
        mach_out: n0.kin.RelMach.Hint,
        beta: n0.kin.FlowAngleRel.Hint,
    ):
        delta0 = aungier_delta0(met_angle)
        delta_sub = aungier_delta_sub(delta0, mach_out)

        deviation_sub = safe_if_else(
            mach_out <= 0.5,
            delta0,
            delta_sub,
        )

        residual = beta - (met_angle - deviation_sub)  # Deviation angle

        return residual


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


# *** Impeller
class ChenSlip(DeviationModel):
    """
    Rotor-inlet flow slip for radial/mixed-flow turbines, from H. Chen and
    N. C. Baines, "The aerodynamic loading of radial and mixed-flow
    turbines," Int. J. Mech. Sci. 36(1), pp. 63-79 (1994), Eq. (8) (eddy
    contained within the blade passage). Meant to be attached directly to
    a rotor ``BladeRow``'s own inlet node (position 0, i.e. ``ChenSlip(): 0``
    in its ``extra_equations``) -- see ``StanitzSlip`` for the alternate
    Stanitz (Eq. 4) correlation.
    """

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        metal_ang0: n0.geo.MetalAngle.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
        n_bl0: n0.geo.NumBlades.Hint,
        wm0: n0.kin.W_mer.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        slip0: n0.oth.SlipFactor.Hint,
    ):
        sin_cone = safe_abs(np.sin(mer_angle0))  # ty: ignore
        cos_metal = np.cos(metal_ang0)
        sin_pitch = np.sin(np.pi / n_bl0)

        # Eq. (8)
        slip_target = (
            1 - (2 * sin_pitch / (np.pi * (1 + sin_pitch))) * sin_cone * cos_metal
        )
        r_slip = slip0 - slip_target

        # Eq. (5), reworked for the actual tangential velocity: ideal
        # relative Wt minus the Wiesner-style slip deficit (1 - mu) * U.
        wt_ideal = wm0 * np.tan(metal_ang0)
        wt_actual = wt_ideal - (1 - slip0) * u0
        r_incidence = wm0 * np.tan(beta0) - wt_actual

        return r_slip, r_incidence


class StanitzSlip(DeviationModel):
    """
    Rotor-inlet flow slip for radial/mixed-flow turbines, from the
    original centrifugal-compressor Stanitz correlation, Eq. (4) in H.
    Chen and N. C. Baines, "The aerodynamic loading of radial and
    mixed-flow turbines," Int. J. Mech. Sci. 36(1), pp. 63-79 (1994).
    Meant to be attached directly to a rotor ``BladeRow``'s own inlet node
    (position 0, i.e. ``StanitzSlip(): 0`` in its ``extra_equations``) --
    see ``ChenSlip`` for the alternate Chen & Baines (Eq. 8) correlation.
    """

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        metal_ang0: n0.geo.MetalAngle.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
        n_bl0: n0.geo.NumBlades.Hint,
        wm0: n0.kin.W_mer.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        slip0: n0.oth.SlipFactor.Hint,
    ):
        sin_cone = safe_abs(np.sin(mer_angle0))  # ty: ignore
        cos_metal = np.cos(metal_ang0)

        # Eq. (4)
        slip_target = 1 - 0.63 * np.pi / n_bl0 * sin_cone * cos_metal
        r_slip = slip0 - slip_target

        # Eq. (5), reworked for the actual tangential velocity: ideal
        # relative Wt minus the Wiesner-style slip deficit (1 - mu) * U.
        wt_ideal = wm0 * np.tan(metal_ang0)
        wt_actual = wt_ideal - (1 - slip0) * u0
        r_incidence = wm0 * np.tan(beta0) - wt_actual

        return r_slip, r_incidence
