"""
Sanity checks for the equation base class in adet.equations.base_equation
"""

from typing import Any

import pytest

from adet.equations.base_equation import EquationBase
from adet.variables import NodeVariables, VarSpec

n0 = NodeVariables(0)
n1 = NodeVariables(1)

dht_test0 = VarSpec('delta_hmass_test', 'J / kg', node=0)
dht_test1 = VarSpec('delta_hmass_test', 'J / kg', node=1)


class DummyEq(EquationBase):
    def residual(
        self,
        v0: n0.kin.V_mag.Hint,
        h0: n0.tot.Enthalpy.Hint,
        h1: n1.tot.Enthalpy.Hint,
        dht0: dht_test0.Hint,
        dht1: dht_test1.Hint,
    ):
        return h0 + h1 + v0 + dht0 + dht1


def _residual(model: Any, *args: Any) -> Any:
    """Evaluate a residual on plain floats (the hints are symbolic types)."""
    return model.residual(*args)


class TestEquationBase:
    def test_arg_symbols_match_residual_signature(self):
        eq = DummyEq()
        assert len(eq.arg_symbols) == 5

    def test_residual_sums_its_arguments(self):
        eq = DummyEq()
        assert _residual(eq, 1.0, 2.0, 3.0, 4.0, 5.0) == pytest.approx(15.0)
