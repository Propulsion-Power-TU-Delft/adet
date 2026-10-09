"""Custom per-node variables shared by the quasi-3D throughflow examples."""

from adet.variables import VariableEnum
from adet.varspec import VarSpec


class MeridionalFlowVariables(VariableEnum):
    """Custom variable for the Eq. (3.6) mass-flow closure."""

    CumFlow = VarSpec('cum_mdot', 'kg / s', 0.5)
    """Mass flow accumulated from the hub up to this spanwise station."""


class BladeLoadingVariables(VariableEnum):
    """Custom variables for the blade-to-blade loading (Section 3.1.2)."""

    W_ss = VarSpec('W_ss', 'm / s', 120.0, (0.0, 2e3))
    """Relative velocity on the suction side, :math:`W_{SS}`."""

    W_ps = VarSpec('W_ps', 'm / s', 80.0, (0.0, 2e3))
    """Relative velocity on the pressure side, :math:`W_{PS}`."""

    DeltaW = VarSpec('delta_W', 'm / s', 20.0, (-1e3, 1e3))
    """Blade loading :math:`W_{SS} - W_{PS}`, Eq. (3.13)."""

    P_ss = VarSpec('p_ss', 'Pa', 1e5, (1e2, 2e7))
    """Static pressure on the suction side."""

    P_ps = VarSpec('p_ps', 'Pa', 1.2e5, (1e2, 2e7))
    """Static pressure on the pressure side."""
