"""Module that gathers fundamental equations for internal flows"""

import numpy as np

from adet.equations import EquationBase
from adet.equations.base_equation import EquationConfig, MeridAreaBlockage
from adet.equations.utils import (
    get_midspan_idx,
    safe_min,
    safe_sum,
    span_fin_diff,
)
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)


class EulerEquation(EquationBase):
    def residual(
        self,
        vt0: n0.kin.V_tan.Hint,
        vt1: n1.kin.V_tan.Hint,
        ht0: n0.tot.Enthalpy.Hint,
        ht1: n1.tot.Enthalpy.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        u1: n1.kin.BladeSpeed.Hint,
    ):
        return (ht1 - ht0) - (u1 * vt1 - u0 * vt0)


class ConstantAngMomentum(EquationBase):
    def residual(
        self,
        rr0: n0.geo.RDistr.Hint,
        rr1: n1.geo.RDistr.Hint,
        vt0: n0.kin.V_tan.Hint,
        vt1: n1.kin.V_tan.Hint,
    ):
        return rr0 * vt0 - rr1 * vt1


class ConstRelEnthalpy(EquationBase):
    def residual(self, h0: n0.rlt.Enthalpy.Hint, h1: n1.rlt.Enthalpy.Hint):
        return h0 - h1


class MassConservation(EquationBase):
    def residual(
        self, mf0: n0.oth.StreamMassFlow.Hint, mf1: n1.oth.StreamMassFlow.Hint
    ):
        return mf0 - mf1


class TotalMassFlow(EquationBase):
    """Cumulative massflow"""

    def residual(
        self, cum_mf0: n0.oth.TotMassFlow.Hint, mf0: n0.oth.StreamMassFlow.Hint
    ):
        return cum_mf0 - safe_sum(mf0)


class LimitedMassflow(EquationBase):
    def residual(
        self,
        mf_target: n0.oth.TargMassFlow.Hint,
        mf_actual: n0.oth.StreamMassFlow.Hint,
        mf_choke: n0.oth.ChokeMassflow.Hint,
    ):

        return mf_actual - safe_min(mf_target, mf_choke)


class TotalArea(EquationBase):
    def residual(self, cum_area0: n0.geo.CumArea.Hint, a0: n0.geo.Area.Hint):
        return cum_area0 - safe_sum(a0)


class MassAreaRelation(EquationBase):
    """
    .. math::
        \\dot_{m} = \\rho_0 V_{m0} A_0
    """

    def residual(
        self,
        vm0: n0.kin.V_mer.Hint,
        a_eff0: n0.geo.EffArea.Hint,
        rho0: n0.stc.Density.Hint,
        mf0: n0.oth.StreamMassFlow.Hint,
    ):
        return mf0 - rho0 * vm0 * a_eff0


class ZeroBlockage(MeridAreaBlockage):
    """Use the annuli's area as the passage area"""

    def residual(self, a0: n0.geo.Area.Hint, a_eff0: n0.geo.EffArea.Hint):
        return a_eff0 - a0


class BladeBlockage(MeridAreaBlockage):
    """Account for blade thicknesses in passage areas"""

    def residual(
        self,
        hh0: n0.geo.HDistr.Hint,
        area0: n0.geo.Area.Hint,
        a_eff0: n0.geo.EffArea.Hint,
        n_blades0: n0.geo.NumBlades.Hint,
        bld_thick0: n0.geo.BldThick.Hint,
        metal_angle0: n0.geo.MetalAngle.Hint,
        disp_thick0: n0.oth.DispThick.Hint,
    ):
        return a_eff0 - (
            area0 - hh0 * n_blades0 * (bld_thick0 + disp_thick0) / np.cos(metal_angle0)
        )


class Kinematics(EquationBase):
    def residual(
        self,
        v0: n0.kin.V_mag.Hint,
        vm0: n0.kin.V_mer.Hint,
        vt0: n0.kin.V_tan.Hint,
        w0: n0.kin.W_mag.Hint,
        wt0: n0.kin.W_tan.Hint,
        wm0: n0.kin.W_mer.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        alpha0: n0.kin.FlowAngleAbs.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        omega0: n0.kin.Omega.Hint,
        rr0: n0.geo.RDistr.Hint,
    ):
        r1 = v0 - (vm0**2 + vt0**2) ** 0.5
        r2 = w0 - (wm0**2 + wt0**2) ** 0.5

        r3 = vm0 - wm0
        r4 = vt0 - (wt0 + u0)

        r5 = alpha0 - np.atan2(vt0, vm0)
        r6 = beta0 - np.atan2(wt0, wm0)

        r7 = omega0 * rr0 - u0

        return r1, r2, r3, r4, r5, r6, r7


class TotalStaticMatching(EquationBase):
    """
    Match the total and static states imposing equal
    entropy and

    .. math::
        h_{t0} = h_0 + \\frac{V_0^2}{2}

    .. math::
        h_{t0}^{rel} = h_0 + \\frac{W_0^2}{2}

    Note
    ----
    The total and relative total entropy do not have a real
    physical sense, as the total and rel. tot. states are defined
    by an isentropic alting of the flow.

    Nonetheless, our formulation treats the
    three states as independent equations of state, which are
    matched in an Equation-Oriented using this equation.

    Observe that mathematically the entropy equality does not affect
    in any way the convergence of the system, and it is immediatly
    satisfied after the first N-R Iteration
    """

    def residual(
        self,
        ht0: n0.tot.Enthalpy.Hint,
        h0: n0.stc.Enthalpy.Hint,
        hr0: n0.rlt.Enthalpy.Hint,
        st0: n0.tot.Entropy.Hint,
        s0: n0.stc.Entropy.Hint,
        sr0: n0.rlt.Entropy.Hint,
        v0: n0.kin.V_mag.Hint,
        w0: n0.kin.W_mag.Hint,
    ):
        r1 = ht0 - (h0 + v0**2 / 2)
        r2 = hr0 - (h0 + w0**2 / 2)
        r3 = st0 - s0
        r4 = sr0 - s0

        return r1, r2, r3, r4


class SimpleRadialEquilibrium(EquationBase):
    """
    Most implementation of a radial equilibrium,
    zero streamline curvature is assumed
    """

    config = EquationConfig(manual_units=('J / kg / m',))

    def residual(
        self,
        rr0: n0.geo.RDistr.Hint,
        p0: n0.stc.Pressure.Hint,
        vt0: n0.kin.V_tan.Hint,
        rho0: n0.stc.Density.Hint,
    ):
        dp_dr = span_fin_diff(p0, rr0)
        return dp_dr / rho0 - vt0**2 / rr0


class NisRe(EquationBase):
    """Non-ISentropic Radial Equilibrium"""

    config = EquationConfig(manual_units=('J / kg / m',))

    def residual(
        self,
        rr0: n0.geo.RDistr.Hint,
        vt0: n0.kin.V_tan.Hint,
        vm0: n0.kin.V_mer.Hint,
        ht0: n0.tot.Enthalpy.Hint,
        T0: n0.stc.Temperature.Hint,
        s0: n0.stc.Entropy.Hint,
    ):
        dVt_dr = span_fin_diff(vt0, rr0)
        dVm_dr = span_fin_diff(vm0, rr0)
        dht_dr = span_fin_diff(ht0, rr0)
        ds_dr = span_fin_diff(s0, rr0)

        lhs = vm0 * dVm_dr + vt0 * dVt_dr + vt0**2 / rr0
        rhs = dht_dr - T0 * ds_dr
        return lhs - rhs


class FreeVortexDistribution(EquationBase):
    def residual(
        self,
        rr0: n0.geo.RDistr.Hint,
        vt0: n0.kin.V_tan.Hint,
        Vm0: n0.kin.V_mer.Hint,
        vm_mid: n0.kin.V_merMid.Hint,
    ):
        midspan = get_midspan_idx(rr0)
        if midspan == 0:
            raise RuntimeError(f'{self} is undefined for single span')

        rVt_mid = rr0[midspan] * vt0[midspan]

        r1 = rr0[:midspan] * vt0[:midspan] - rVt_mid
        r2 = rr0[midspan + 1 :] * vt0[midspan + 1 :] - rVt_mid
        r3 = Vm0[:midspan] - vm_mid
        r4 = Vm0[midspan + 1 :] - vm_mid
        r5 = vm_mid - Vm0[midspan]
        return r1, r2, r3, r4, r5


class ForcedVortexDistribution(EquationBase):
    def residual(self, rr0: n0.geo.RDistr.Hint, vt0: n0.kin.V_tan.Hint):
        midspan = get_midspan_idx(rr0)
        Vt_by_r_mid = vt0[midspan] / rr0[midspan]

        r1 = vt0[:midspan] / rr0[:midspan] - Vt_by_r_mid
        r2 = vt0[midspan + 1 :] / rr0[midspan + 1 :] - Vt_by_r_mid
        return r1, r2


class AxialMomentumBalance(EquationBase):
    def residual(
        self,
        v0: n0.kin.W_mag.Hint,
        rho0: n0.stc.Density.Hint,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        p0: n0.stc.Pressure.Hint,
        v1: n1.kin.W_mag.Hint,
        rho1: n1.stc.Density.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        p1: n1.stc.Pressure.Hint,
        pitch0: n0.geo.Pitch.Hint,
        metal_angle: n0.geo.MetalAngle.Hint,
        delta_s: n1.loss.Ds_mixing.Hint,
    ):
        throat = pitch0 * np.cos(metal_angle)
        dev = beta0 - beta1

        mom_inl = p0 * throat + rho0 * throat * v0 * v0
        mom_out = p1 * throat + rho1 * throat * v0 * v1 * np.cos(dev)

        r1 = mom_inl - mom_out
        r2 = delta_s - (s1 - s0)

        return r1, r2
