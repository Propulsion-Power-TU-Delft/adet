"""
Module that gathers equations that represent definitions of nondimensional
coefficients used in TurboMachinery
"""

import types
from typing import Sequence

import CoolProp as cp
import numpy as np

from adet.equations import EquationBase
from adet.equations.base_equation import EquationConfig
from adet.equations.utils import get_midspan_idx, safe_abs, thermo_deriv
from adet.variables import NodeVariables, ThermoVariables, VarSpec

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)
thrm = ThermoVariables()


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
    Static-to-total pressure ratio (stagnation pressure to static pressure)

    .. math::
        \\frac{p_1}{p_{t0}} = \\text{PRatioTS}
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
    Volumetric flow ratio at midspan (density ratio)

    .. math::
        \\text{VolflowRatio} = \\frac{\\rho_{t0}}{\\rho_1} \\bigg|_{\\text{mid}}
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
        out_properties=(thrm.Enthalpy,),
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


class TotalStaticExpansionEfficiency(EquationBase):
    """Total-to-static isentropic expansion efficiency (cf. Chen & Baines
    1994, Eq. 10):

    .. math::
        \\eta_{ts} = \\frac{h_{t0} - h_{t1}}{h_{t0} - h_{1,s}}

    :math:`h_{1,s}` is the enthalpy at the actual static exit pressure and
    the inlet entropy.
    """

    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(thrm.Enthalpy,),
    )

    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h0_tot: n0.tot.Enthalpy.Hint,
        h1_tot: n1.tot.Enthalpy.Hint,
        eta_ts1: n1.ndim.EtaTS.Hint,
    ):
        h_is1 = self.eos(p1, s0)
        return eta_ts1 - (h0_tot - h1_tot) / (h0_tot - h_is1)


def _zeta_spec(ds_spec: VarSpec) -> VarSpec:
    return VarSpec(
        f'zeta_{ds_spec.symbol.removeprefix("ds_")}',
        '',
        guess=0.05,
        bounds=(0.0, 1.0),
        node=0,
    )


class LossCoefficient(EquationBase):
    """Denton-style loss coefficient, one per loss mechanism plus their
    sum:

    .. math::
        \\zeta_i = \\frac{T_{is} \\cdot \\Delta s_i}{W_{is}^2 / 2}
        \\qquad
        \\zeta_{total} = \\sum_i \\zeta_i

    All at the same node, where :math:`T_{is}`/:math:`W_{is}` are the
    isentropic static temperature/relative velocity computed by
    `IsentropicProperties` (`equations/definitions.py`).

    `ds_specs` is normally obtained from
    `losses.base_loss.collect_loss_mechanisms` rather than listed by hand,
    so the number of residuals this equation adds follows directly from
    which `LossModel` equations were actually added at that node -- one
    ``zeta_<mechanism>`` unknown/residual per mechanism Ds, plus
    ``zeta_total``. Add this equation at a single node (the component's
    outlet).
    """

    def __init__(
        self, ds_specs: Sequence[VarSpec], scaling_factor: list[float] | None = None
    ):
        self.ds_specs = [spec.Glob.at_node(0) for spec in ds_specs]
        self.zeta_specs = [_zeta_spec(spec) for spec in self.ds_specs]
        self.zeta_total_spec = VarSpec(
            'zeta_total', '', guess=0.1, bounds=(-1.0, 1.0), node=0
        )

        super().__init__(scaling_factor)
        self._bind_residual()

    def residual(self, *args):
        """Replaced in `__init__` by `_bind_residual`, based on the
        mechanisms passed in."""
        raise NotImplementedError

    def _bind_residual(self):
        ds_names = [f'ds_{i}' for i in range(len(self.ds_specs))]
        zeta_names = [f'zeta_{i}' for i in range(len(self.zeta_specs))]

        if ds_names:
            # `t_is`/`w_is` are only pulled in (and so only need to be
            # solved elsewhere, by `IsentropicProperties`) when there is
            # at least one mechanism to weigh by them -- a no-mechanism
            # component (e.g. an isentropic interspace) never needs them.
            params = ['self', 't_is', 'w_is', *ds_names, *zeta_names, 'zeta_total']
            lines = [f'def _residual({", ".join(params)}):']
            for ds_name, zeta_name in zip(ds_names, zeta_names):
                lines.append(
                    f'    r_{zeta_name} = {zeta_name} - '
                    f't_is * {ds_name} / (0.5 * w_is**2)'
                )
            sum_expr = ' + '.join(zeta_names)
            lines.append(f'    r_total = zeta_total - ({sum_expr})')
            return_names = [f'r_{name}' for name in zeta_names] + ['r_total']
        else:
            params = ['self', 'zeta_total']
            lines = [
                f'def _residual({", ".join(params)}):',
                '    r_total = zeta_total - 0.0 * zeta_total',
            ]
            return_names = ['r_total']
        lines.append(f'    return {", ".join(return_names)}')

        namespace: dict = {}
        exec('\n'.join(lines), namespace)
        fn = namespace['_residual']
        annotations = {
            **{name: spec.Hint for name, spec in zip(ds_names, self.ds_specs)},
            **{name: spec.Hint for name, spec in zip(zeta_names, self.zeta_specs)},
            'zeta_total': self.zeta_total_spec.Hint,
        }
        if ds_names:
            annotations['t_is'] = n0.oth.Tis_stc.Hint
            annotations['w_is'] = n0.oth.W_Is.Hint
        fn.__annotations__ = annotations
        self.residual = types.MethodType(fn, self)


class EfficiencyDeficit(EquationBase):
    """Efficiency deficit attributable to each loss mechanism, plus their
    sum:

    .. math::
        \\Delta\\eta_i = \\zeta_i \\cdot \\frac{W_{is}^2 / 2}{work_{is}}
        \\cdot 100
        \\qquad
        \\Delta\\eta_{total} = \\sum_i \\Delta\\eta_i

    **Depends on `LossCoefficient` already being added at the same node
    with the same `ds_specs`**: this equation reads its ``zeta_i``/
    ``zeta_total`` as already-solved inputs rather than recomputing them,
    so add `LossCoefficient` first.

    :math:`work_{is}` is the isentropic work of the whole expansion/
    compression (e.g. `IsentropicEulerWork`, `equations/definitions.py`),
    which generally lives at a different (machine-reference) node than
    this component's own outlet -- hence this equation spans two nodes:
    the component's outlet (node 0) and the machine-reference node
    (node 1).
    """

    def __init__(
        self, ds_specs: Sequence[VarSpec], scaling_factor: list[float] | None = None
    ):
        ds_specs = [spec.Glob.at_node(0) for spec in ds_specs]
        self.zeta_specs = [_zeta_spec(spec) for spec in ds_specs]
        self.zeta_total_spec = VarSpec(
            'zeta_total', '', guess=0.1, bounds=(-1.0, 1.0), node=0
        )
        self.effdef_specs = [
            VarSpec(
                f'effdef_{spec.symbol.removeprefix("ds_")}',
                '',
                guess=1.0,
                bounds=(-100.0, 100.0),
                node=0,
            )
            for spec in ds_specs
        ]
        self.effdef_total_spec = VarSpec(
            'effdef_total', '', guess=5.0, bounds=(0.0, 100.0), node=0
        )

        super().__init__(scaling_factor)
        self._bind_residual()

    def residual(self, *args):
        """Replaced in `__init__` by `_bind_residual`, based on the
        mechanisms passed in."""
        raise NotImplementedError

    def _bind_residual(self):
        zeta_names = [f'zeta_{i}' for i in range(len(self.zeta_specs))]
        effdef_names = [f'effdef_{i}' for i in range(len(self.effdef_specs))]

        if zeta_names:
            # `w_is`/`work_is` are only pulled in when there is at least
            # one mechanism to weigh by them -- with none, `zeta_total` is
            # already pinned to 0 by `LossCoefficient`, so `effdef_total`
            # can just as trivially follow without needing either.
            params = [
                'self',
                'w_is',
                'work_is',
                *zeta_names,
                *effdef_names,
                'zeta_total',
                'effdef_total',
            ]
            lines = [
                f'def _residual({", ".join(params)}):',
                '    ideal_ke = 0.5 * w_is**2',
            ]
            for zeta_name, effdef_name in zip(zeta_names, effdef_names):
                lines.append(
                    f'    r_{effdef_name} = {effdef_name} - '
                    f'{zeta_name} * ideal_ke / work_is * 100'
                )
            lines.append(
                '    r_total = effdef_total - zeta_total * ideal_ke / work_is * 100'
            )
            return_names = [f'r_{name}' for name in effdef_names] + ['r_total']
        else:
            params = ['self', 'zeta_total', 'effdef_total']
            lines = [
                f'def _residual({", ".join(params)}):',
                '    r_total = effdef_total - 0.0 * zeta_total',
            ]
            return_names = ['r_total']
        lines.append(f'    return {", ".join(return_names)}')

        namespace: dict = {}
        exec('\n'.join(lines), namespace)
        fn = namespace['_residual']
        annotations = {
            **{name: spec.Hint for name, spec in zip(zeta_names, self.zeta_specs)},
            **{name: spec.Hint for name, spec in zip(effdef_names, self.effdef_specs)},
            'zeta_total': self.zeta_total_spec.Hint,
            'effdef_total': self.effdef_total_spec.Hint,
        }
        if zeta_names:
            annotations['w_is'] = n0.oth.W_Is.Hint
            annotations['work_is'] = n1.oth.EulerWorkIs.Hint
        fn.__annotations__ = annotations
        self.residual = types.MethodType(fn, self)


class TotalTotalCompressionEfficiency(EquationBase):
    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(thrm.Enthalpy,),
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
        out_properties=(thrm.Enthalpy,),
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

        return delta_h_tot_stage[midspan] * react_ts3 - delta_h_rotor[midspan]


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
        return safe_abs(u1) * flow_coeff1 - vm0


class FlowCoefficientMid(EquationBase):
    """
    .. math::
        \\phi = \\frac{V_{m0}}{U_{0}}
    """

    def residual(
        self,
        vm0: n0.kin.V_mer.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        flow_coeff1: n1.ndim.FlowCoeffMid.Hint,
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
        return u1**2 * work_coeff1 - (h1_tot - h0_tot)


class WorkCoefficientMid(EquationBase):
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
        work_coeff1: n1.ndim.WorkCoeffMid.Hint,
    ):
        midspan = get_midspan_idx(h0_tot)

        return u1[midspan] ** 2 * work_coeff1 - (h1_tot[midspan] - h0_tot[midspan])


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
        mf0: n0.oth.StreamMassFlow.Hint,
    ):
        return swll_cap0 - mf0 / (rho0_tot * u1 * (2 * rr1) ** 2)


class SpecificSpeed(EquationBase):
    def residual(
        self,
        spec_speed1: n1.ndim.SpecificSpeed.Hint,
        omega1: n1.kin.Omega.Hint,
        mf1: n1.oth.StreamMassFlow.Hint,
        rho1: n1.stc.Density.Hint,
        h0_tot: n0.tot.Enthalpy.Hint,
        h1: n1.stc.Enthalpy.Hint,
    ):
        return spec_speed1 * ((h0_tot - h1) ** (3 / 4)) - omega1 * np.sqrt(mf1 / rho1)


class SizeParameter(EquationBase):
    def residual(
        self,
        size_param1: n1.ndim.SizeParameter.Hint,
        mf1: n1.oth.StreamMassFlow.Hint,
        rho1: n1.stc.Density.Hint,
        h0_tot: n0.tot.Enthalpy.Hint,
        h1: n1.stc.Enthalpy.Hint,
    ):
        return size_param1 * ((h0_tot - h1) ** (1 / 4)) - (mf1 / rho1) ** 0.5


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
        w0: n0.kin.W_mag.Hint,
        a_sound0: n0.stc.SpeedSound.Hint,
        rel_mach0: n0.kin.RelMach.Hint,
    ):
        return rel_mach0 * a_sound0 - w0


class GammaPV(EquationBase):
    config = EquationConfig(
        input_pair=cp.DmassSmass_INPUTS,
        out_properties=(thrm.Pressure,),
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


class GammaIdeal(EquationBase):
    def residual(
        self,
        cp0: n0.stc.Cp.Hint,
        cv0: n0.stc.Cv.Hint,
        gamma_pv0: n0.oth.GammaPV.Hint,
    ):
        return gamma_pv0 - cp0 / cv0
