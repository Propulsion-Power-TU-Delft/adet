"""
Sanity checks for the general-purpose loss models in adet.losses.basic
"""

from typing import Any

import pytest

from adet.losses.basic import (
    IsentropicLink,
    PercentageEntropyLoss,
    PercTotalPressureLoss,
    TotalPressureLoss,
)


def _residual(model: Any, *args: Any) -> Any:
    """Evaluate a loss residual on plain floats (the hints are symbolic types)."""
    return model.residual(*args)


class TestPercTotalPressureLoss:
    def test_residual_zero_at_expected_outlet_pressure(self):
        loss = PercTotalPressureLoss(loss_coefficient=0.1)
        # p1 = p0 * (1 - C) = 90
        assert _residual(loss, 100.0, 90.0) == pytest.approx(0.0)

    def test_residual_nonzero_away_from_solution(self):
        loss = PercTotalPressureLoss(loss_coefficient=0.1)
        assert _residual(loss, 100.0, 80.0) != pytest.approx(0.0)

    def test_zero_coefficient_is_identity(self):
        loss = PercTotalPressureLoss()
        assert _residual(loss, 100.0, 100.0) == pytest.approx(0.0)


class TestTotalPressureLoss:
    def test_residual_zero_at_expected_outlet_pressure(self):
        # Y = (pt0 - pt1) / (pt0 - p0) = 0.2 -> pt1 = pt0 - Y * (pt0 - p0) = 98
        loss = TotalPressureLoss(loss_coefficient=0.2)
        assert _residual(loss, 100.0, 90.0, 98.0) == pytest.approx(0.0)

    def test_residual_value(self):
        loss = TotalPressureLoss(loss_coefficient=0.2)
        # (100 - 92) - (100 - 90) * 0.2 = 6
        assert _residual(loss, 100.0, 90.0, 92.0) == pytest.approx(6.0)

    def test_no_loss_when_outlet_equals_inlet_total(self):
        loss = TotalPressureLoss(loss_coefficient=0.0)
        assert _residual(loss, 100.0, 90.0, 100.0) == pytest.approx(0.0)


class TestIsentropicLink:
    def test_residual_zero_for_equal_entropy(self):
        link = IsentropicLink()
        assert _residual(link, 2.0, 2.0) == pytest.approx(0.0)

    def test_residual_nonzero_for_entropy_increase(self):
        link = IsentropicLink()
        assert _residual(link, 2.0, 2.1) == pytest.approx(0.1)


class TestPercentageEntropyLoss:
    def test_residual_zero_at_expected_outlet_entropy(self):
        # s1 = s0 * (1 + 0.05) = 2.1
        loss = PercentageEntropyLoss(entropy_generation=0.05)
        assert _residual(loss, 2.0, 2.1) == pytest.approx(0.0)

    def test_residual_nonzero_away_from_solution(self):
        loss = PercentageEntropyLoss(entropy_generation=0.05)
        assert _residual(loss, 2.0, 2.0) != pytest.approx(0.0)

    def test_zero_generation_is_isentropic(self):
        loss = PercentageEntropyLoss()
        assert _residual(loss, 2.0, 2.0) == pytest.approx(0.0)
