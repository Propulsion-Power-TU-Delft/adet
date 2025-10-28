from adet.losses import LossModel


class PercentageEntropyLoss(LossModel):
    definition = 'delta_smass'
    identifier = 'pct'

    def __init__(self, percentage_loss: float, scaling_factor=None):
        parameters = {
            'percentage_loss': percentage_loss,
        }
        super().__init__(scaling_factor, **parameters)

    def value(self, stc_smass0, oth_percentage_loss1):
        return oth_percentage_loss1 * stc_smass0


if __name__ == '__main__':
    loss_instance = PercentageEntropyLoss(0.0)
