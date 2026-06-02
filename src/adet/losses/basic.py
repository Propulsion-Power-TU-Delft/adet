from adet.variables import NodeVariables
from adet.equations.base_equation import DeviationModel, LossApplier
from adet.equations.utils import get_midspan_idx

n0 = NodeVariables(0)
n1 = NodeVariables(1)


class PercTotalPressureLoss(LossApplier):
    """
    .. math::

        \\mathrm{Y} = \\frac{p_{t1}^{r}- p_{t0}^{r}}{p_{t0}^{r} - p_0}
    """

    def __init__(
        self, loss_coefficient: float = 0.0, scaling_factor: list[float] | None = None
    ):
        super().__init__(scaling_factor)
        self.loss_coeff = loss_coefficient

    def residual(self, rlt_p0, rlt_p1):
        return rlt_p1 - rlt_p0 * (1 - self.loss_coeff)


class TotalPressureLoss(LossApplier):
    """
    .. math::

        \\mathrm{Y} = \\frac{p_{t1}^{r}- p_{t0}^{r}}{p_{t0}^{r} - p_0}
    """

    def __init__(self, loss_coefficient: float = 0.0):
        super().__init__()
        self.loss_coeff = loss_coefficient

    def residual(
        self,
        rlt_p0: n0.rlt.Pressure.Hint,
        stc_p0: n0.stc.Pressure.Hint,
        rlt_p1: n1.rlt.Pressure.Hint,
    ):
        return (rlt_p0 - rlt_p1) - (rlt_p0 - stc_p0) * self.loss_coeff


class ThroatLossCoefficient(LossApplier):
    def __init__(self, loss_coefficient: float = 0.0):
        super().__init__()
        self.loss_coeff = loss_coefficient

    def residual(
        self,
        stc_p1: n1.stc.Pressure.Hint,
        rlt_p1: n1.rlt.Pressure.Hint,
        U0: n0.kin.BladeSpeed.Hint,
        htr0: n0.rlt.Enthalpy.Hint,
        s0: n0.stc.Entropy.Hint,
    ):
        htr1_is = roth0 - U0
        ptr0 = self.eos(htr0, s0)

        return (rlt_p1_is - rlt_p1) - (rlt_p1 - stc_p1) * self.loss_coeff


class PlaceHolderLoss(LossApplier):
    """
    Use when defining efficiency through eta_tt for example instead of direct
    row-based loss coefficients. This is because I made components raise
    errors when they are missing loss models, since I always forget them
    """

    def residual(self, stc_smass0, stc_smass1):
        return ()


class IsentropicLink(LossApplier):
    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
    ):
        return s1 - s0


class PercentageEntropyLoss(LossApplier):
    """
    Percentage increase of entropy w.r.t. the inlet conditions

    Parameters
    ----------
    entropy_generation: float = 0.05
        Relative increase of entropy (e.g. 0.05 -> 5% entropy increase)

    .. math::
        s_{1} = s_{0} \\cdot (1 + \\mathrm{C})
    """

    def __init__(self, entropy_generation: float = 0.0, scaling_factor=None):
        super().__init__(scaling_factor)
        self.entropy_gen = entropy_generation

    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
    ):
        return s1 - s0 * (1 + self.entropy_gen)


class FixedEnthalpyLoss(LossApplier):
    def __init__(self, enthalpy_generated: float = 0.0):
        super().__init__()
        self.enth_gen = enthalpy_generated

    def residual(self, tot_hmass0, oth_tot_hmass_is0, oth_delta_tot_hmass0):
        # Actual enthalpy = outlet isentropic + generated
        return tot_hmass0 - (oth_tot_hmass_is0 - oth_delta_tot_hmass0)


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
