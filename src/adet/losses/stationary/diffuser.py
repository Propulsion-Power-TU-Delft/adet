"""Stationary component loss models: vaneless and vaned diffusers."""

import CoolProp as cp
import numpy as np

from adet.equations.base_equation import (
    EquationBase,
    EquationConfig,
)
from adet.equations.utils import (
    safe_max,
)
from adet.losses.base_loss import LossModel
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


class AmiranteDiffuserMomentum(EquationBase):
    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(thrm.Enthalpy,),
        manual_units=('m^2 / s', 'K'),
    )

    def residual(
        self,
        alpha1: n1.kin.FlowAngleAbs.Hint,
        hgt1: n1.geo.Height.Hint,
        rr1: n1.geo.RDistr.Hint,
        rr0: n0.geo.RDistr.Hint,
        v0: n0.kin.V_mag.Hint,
        v1: n1.kin.V_mag.Hint,
        vt0: n0.kin.V_tan.Hint,
        p0: n0.stc.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        rho1: n1.stc.Density.Hint,
        cp1: n1.stc.Cp.Hint,
        cv1: n1.stc.Cv.Hint,
        T0: n0.stc.Temperature.Hint,
        T1: n1.stc.Temperature.Hint,
        vt1: n1.kin.V_tan.Hint,
        visc1: n1.stc.Viscosity.Hint,
        wake_frac0: n0.oth.WakeFrac.Hint,
    ):
        FRIC_CONST = 0.02
        ETA_POLY = 0.85

        delta_rad = safe_max(0.001 * rr0, rr1 - rr0)
        x_log = delta_rad / np.cos(alpha1)
        Re = (rho1 * v1 * x_log) / visc1
        Cf = FRIC_CONST * (1.8e5 / Re) ** 0.2
        Wf = (Cf * (v1 * rr1) ** 2 * delta_rad) / (hgt1 * rr0 * rr1 * np.cos(alpha1))

        r1 = rr0 * vt0 - rr1 * vt1 * (1 + Wf / (wake_frac0 * v1 * v0))

        p_ratio = p1 / p0
        gamma = cp1 / cv1

        exponent = (gamma - 1) / (ETA_POLY * gamma)

        r2 = T1 - T0 * (p_ratio) ** exponent

        return r1, r2


class StanitzDiffuserMomentum(EquationBase):
    """
    Dubitsky and Japikse (2008) - Vaneless Diffuser Advanced Model, Eqs. (1)-(2),
    adapted from Stanitz (1952) - NACA TN 2610, Eqs. (8)-(9).

    Meridional and tangential momentum of the one-zone vaneless diffuser model
    over one radial step between the inlet (0) and outlet (1) of the component.
    Valid for radial and mixed-flow diffusers. The two-zone (jet/wake) model of
    Dubitsky and Japikse, Eqs. (7)-(22), is not implemented: the authors use it
    only together with a two-zone impeller model (not coded into adet).

    Notation, Dubitsky and Japikse -> adet:
    - C, C_m, C_theta -> V_mag, V_mer, V_tan
    - alpha           -> FlowAngleAbs, from the meridional direction
    - phi             -> MeridionalAngle, diffuser inclination, 90 deg = radial
    - b               -> Height, passage depth
    - C_f             -> DiffFricCoeff, wall shear C_f rho C^2 / 2 on both walls

    Discretisation: backward difference over the step, with the
    coefficients evaluated at the outlet of the step (first order), multiplied
    by dr.

    - r1: meridional momentum, Eq. (1), multiplied by dr
    - r2: tangential momentum, Eq. (2), multiplied by dr

    Note: C_f is an input, given at the inlet of the step, either as a boundary
    condition or from DiffuserFrictionCoefficient, Eq. (6).
    """

    def residual(
        self,
        vm0: n0.kin.V_mer.Hint,
        vm1: n1.kin.V_mer.Hint,
        vt0: n0.kin.V_tan.Hint,
        vt1: n1.kin.V_tan.Hint,
        v1: n1.kin.V_mag.Hint,
        p0: n0.stc.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        rho1: n1.stc.Density.Hint,
        rr0: n0.geo.RDistr.Hint,
        rr1: n1.geo.RDistr.Hint,
        hgt1: n1.geo.Height.Hint,
        mer_angle1: n1.geo.MeridionalAngle.Hint,
        cf0: n0.oth.DiffFricCoeff.Hint,
    ):
        dr = rr1 - rr0
        wall1 = hgt1 * np.sin(mer_angle1)

        r1 = (
            vm1 * (vm1 - vm0)
            - vt1**2 / rr1 * dr
            + cf0 * v1 * vm1 / wall1 * dr
            + (p1 - p0) / rho1
        )

        r2 = vm1 * (vt1 - vt0) + vm1 * vt1 / rr1 * dr + cf0 * v1 * vt1 / wall1 * dr

        return r1, r2


class DiffuserFrictionCoefficient(EquationBase):
    """
    Dubitsky and Japikse (2008) - Vaneless Diffuser Advanced Model, Eq. (6),
    attributed there to Japikse (1996). Skin friction coefficient of the
    vaneless diffuser, C_f in StanitzDiffuserMomentum.

    The Reynolds number is not defined in the paper Re = m_dot / (mu D_hyd),
    with D_hyd the hydraulic diameter of the impeller exit passage given as
    an input.

    - r1: friction coefficient, Eq. (6)
    """

    def residual(
        self,
        cf0: n0.oth.DiffFricCoeff.Hint,
        k_diff0: n0.oth.DiffFricConst.Hint,
        cum_mf0: n0.oth.TotMassFlow.Hint,
        visc0: n0.stc.Viscosity.Hint,
        hyd_diam0: n0.geo.HydDiam.Hint,
    ):
        reynolds = cum_mf0 / (visc0 * hyd_diam0)

        return cf0 - k_diff0 * (1.8e5 / reynolds) ** 0.2


class JansenDiffuserLoss(LossModel):
    config = EquationConfig(
        manual_units=('dimensionless', 'J/kg'),
    )

    def residual(
        self,
        rr0: n0.geo.Rmid.Hint,
        rr1: n1.geo.Rmid.Hint,
        hh0: n0.geo.Height.Hint,
        hh1: n1.geo.Height.Hint,
        rho0: n0.stc.Density.Hint,
        vt0: n0.kin.V_tan.Hint,
        v0: n0.kin.V_mag.Hint,
        alpha0: n0.kin.FlowAngleAbs.Hint,
        vt1: n1.kin.V_tan.Hint,
        cum_mf: n0.oth.TotMassFlow.Hint,
        visc0: n0.stc.Viscosity.Hint,
        visc1: n1.stc.Viscosity.Hint,
        gas_const: n0.stc.GasConstant.Hint,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
    ):
        h_mean = (hh0 + hh1) / 2
        visc_mean = (visc0 + visc1) / 2
        hyd_D = 2 * h_mean

        Re = cum_mf / (visc_mean * hyd_D)
        FRIC_CONST = 0.01
        Cf = FRIC_CONST * (1.8e5 / Re) ** 0.2

        r1 = (
            vt0 / vt1
            - rr1 / rr0
            - (2 * np.pi * Cf * rho0 * vt0 * (rr1**2 - rr0 * rr1)) / cum_mf
        )

        loss = (Cf * (1 - (rr0 / rr1) ** 1.5) * v0**2) / (1.5 * hh0 * np.cos(alpha0))

        r2 = s1 - (s0 + np.exp(-loss / gas_const))

        return r1, r2
