from adet.equations.base_equation import DeviationModel, LossApplier
from adet.equations.utils import get_midspan_idx


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

    def residual(self, rlt_p0, stc_p0, rlt_p1):
        return (rlt_p0 - rlt_p1) - (rlt_p0 - stc_p0) * self.loss_coeff


class PlaceHolderLoss(LossApplier):
    """
    Use when defining efficiency through eta_tt for example instead of direct
    row-based loss coefficients. This is because I made components raise
    errors when they are missing loss models, since I always forget them
    """

    def residual(self, stc_smass0, stc_smass1):
        return ()


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

    def residual(self, stc_smass0, stc_smass1):
        return stc_smass1 - stc_smass0 * (1 + self.entropy_gen)


class FixedEnthalpyLoss(LossApplier):
    def __init__(self, enthalpy_generated: float = 0.0):
        super().__init__()
        self.enth_gen = enthalpy_generated

    def residual(self, tot_hmass0, oth_tot_hmass_is0, oth_delta_tot_hmass0):
        # Actual enthalpy = outlet isentropic + generated
        return tot_hmass0 - (oth_tot_hmass_is0 - oth_delta_tot_hmass0)


class ZeroDeviation(DeviationModel):
    """Impose equality between kinematic and geometric angles at a node"""

    def residual(self, geo_metal_angle0, kin_beta0):
        return kin_beta0 - geo_metal_angle0


class ZeroMidspanDeviation(DeviationModel):
    def residual(self, kin_beta0, geo_metal_angle0):
        midspan = get_midspan_idx(kin_beta0)
        return kin_beta0[midspan] - geo_metal_angle0[midspan]
