"""
Very basic loss models in equation form, mainly used for testing and debugging,
the actual loss implementations are to be found in the `losses` module
"""

from adet.equations import EquationBase


class FixedPressureLoss(EquationBase):
    """
    .. math::

        \\mathrm{Y} = \\frac{p_{t1}^{r}- p_{t0}^{r}}{p_{t0}^{r} - p_0}
    """

    def __init__(self, loss_coefficient: float = 0.95):
        super().__init__()
        self.loss_coeff = loss_coefficient

    def _compute_residual(self, rlt_p0, stc_p0, rlt_p1):
        return (rlt_p1 - rlt_p0) - (rlt_p0 - stc_p0) * self.loss_coeff


class PercentageEntropyLoss(EquationBase):
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

    def _compute_residual(self, stc_smass0, stc_smass1):
        return stc_smass1 - stc_smass0 * (1 + self.entropy_gen)


class FixedEnthalpyLoss(EquationBase):
    def _compute_residual(self, tot_hmass0, oth_htis0, oth_ent_loss0):
        return tot_hmass0 - (oth_htis0 + oth_ent_loss0)
