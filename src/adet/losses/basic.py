from adet.equations.base_equation import DeviationModel, LossApplier
from adet.equations.utils import get_midspan_idx
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)


class PercTotalPressureLoss(LossApplier):
    """
    .. math::

        p_{t1,r} = p_{t0,r} (1 - \\mathrm{C})

    Parameters
    ----------
    loss_coefficient: float
        Loss coefficient as defined above
    """

    def __init__(self, loss_coefficient: float = 0.0):
        self.loss_coeff = loss_coefficient

    def residual(
        self,
        rlt_p0: n0.rlt.Pressure.Hint,
        rlt_p1: n1.rlt.Pressure.Hint,
    ):
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

    .. math::
        s_{1} = s_{0} \\cdot (1 + \\mathrm{C})

    Parameters
    ----------
    entropy_generation: float = 0.05
        Relative increase of entropy (e.g. 0.05 -> 5% entropy increase)

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
