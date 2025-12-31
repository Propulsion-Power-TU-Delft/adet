from adet.equations.base_equation import DeviationModel
from adet.losses import LossModel


class PercTotalPressureLoss(LossModel):
    """
    .. math::

        \\mathrm{Y} = \\frac{p_{t1}^{r}- p_{t0}^{r}}{p_{t0}^{r} - p_0}
    """

    def __init__(self, loss_coefficient: float = 0.0):
        super().__init__()
        self.loss_coeff = loss_coefficient

    def residual(self, rlt_p0, rlt_p1):
        return rlt_p1 - rlt_p0 * (1 - self.loss_coeff)


class TotalPressureLoss(LossModel):
    """
    .. math::

        \\mathrm{Y} = \\frac{p_{t1}^{r}- p_{t0}^{r}}{p_{t0}^{r} - p_0}
    """

    def __init__(self, loss_coefficient: float = 0.0):
        super().__init__()
        self.loss_coeff = loss_coefficient

    def residual(self, rlt_p0, stc_p0, rlt_p1):
        return (rlt_p0 - rlt_p1) - (rlt_p0 - stc_p0) * self.loss_coeff


class PlaceHolderLoss(LossModel):
    """
    Use when defining efficiency through eta_tt for example instead of direct
    row-based loss coefficients
    """

    def residual(self, stc_smass0, stc_smass1):
        return ()


class PercentageEntropyLoss(LossModel):
    """
    Percentage increase of entropy w.r.t. the inlet conditions

    Parameters
    ----------
    entropy_generation: float = 0.05
        Relative increase of entropy (e.g. 0.05 -> 5% entropy increase)

    .. math::
        s_{1} = s_{0} \\cdot (1 + \\mathrm{C})
    """

    def __init__(self, entropy_generation: float = 0.05):
        super().__init__()
        self.entropy_gen = entropy_generation

    def residual(self, stc_smass0, stc_smass1):
        return stc_smass1 - stc_smass0 * (1 + self.entropy_gen)


class FixedEnthalpyLoss(LossModel):
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
    def residual(
        self,
        kin_beta0,
        kin_beta_midspan0,
        geo_metal_angle0,
        geo_metal_angle_midspan0,
    ):
        num_span = max(kin_beta0.shape)
        if num_span == 1:
            midspan = 0
        else:
            midspan = num_span // 2

        r1 = kin_beta_midspan0 - kin_beta0[midspan]
        r2 = geo_metal_angle_midspan0 - geo_metal_angle0[midspan]
        r3 = kin_beta_midspan0 - geo_metal_angle_midspan0

        return r1, r2, r3
