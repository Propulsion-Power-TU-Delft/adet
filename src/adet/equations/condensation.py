"""
Equations for humid-air flows with equilibrium condensation, to be used with a
compositional fluid (:class:`adet.fluid.humid_mixture.HumidAirMixture`) in any
``CasadiSystem`` or component network.

Node composition: ``n.mix.HumidityRatio`` (omega = m_v / m_a) and
``n.mix.Wetness`` (beta = m_l / (m_l + m_v)), shared by all node states.

Typical use
-----------
- No phase change between two nodes (dry or frozen flow):
  :class:`FrozenComposition` (2 residuals).
- Equilibrium condensation between two nodes:
  :class:`EquilibriumComposition` (water conservation plus saturation at
  the downstream node, 2 residuals), or equivalently its building blocks
  :class:`WaterConservation` and :class:`PhaseEquilibrium`.
- Thermal choking on the equilibrium speed of sound (Young, 1995):
  :class:`EquilibriumMachNumber` and a boundary condition on
  ``n.mix.EqMach``.

``FrozenComposition`` and ``EquilibriumComposition`` belong to the
``CompositionModel`` family of unique equations: in a ``ComponentNetwork``
with a compositional fluid every component gets ``FrozenComposition`` by
default, overridden by passing e.g.
``extra_equations={EquilibriumComposition(): (0, 1)}`` to the component.

The mass, momentum and energy balances across a condensation front are the
usual ones (e.g. ``MassConservation``, :class:`ConstantAreaMomentum`,
``EulerEquation``): mixture density and enthalpy already include the liquid.
"""

from adet.equations import EquationBase
from adet.equations.base_equation import CompositionModel
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)


class FrozenComposition(CompositionModel):
    """No phase change between two nodes: composition is transported unchanged."""

    def residual(
        self,
        om0: n0.mix.HumidityRatio.Hint,
        om1: n1.mix.HumidityRatio.Hint,
        beta0: n0.mix.Wetness.Hint,
        beta1: n1.mix.Wetness.Hint,
    ):
        return om1 - om0, beta1 - beta0


class EquilibriumComposition(CompositionModel):
    """
    Equilibrium condensation/evaporation between two nodes: total water per
    kg of dry air is conserved and the downstream node is saturated
    (``phi1 = 1``, vapour in equilibrium with its liquid).
    """

    def residual(
        self,
        om0: n0.mix.HumidityRatio.Hint,
        om1: n1.mix.HumidityRatio.Hint,
        beta0: n0.mix.Wetness.Hint,
        beta1: n1.mix.Wetness.Hint,
        phi1: n1.stc.RelHumidity.Hint,
    ):
        return om0 * (1 - beta1) - om1 * (1 - beta0), phi1 - 1


class WaterConservation(EquationBase):
    """
    Total water per kg of dry air is conserved between two nodes:
    ``omega0 / (1 - beta0) = omega1 / (1 - beta1)``.
    """

    def residual(
        self,
        om0: n0.mix.HumidityRatio.Hint,
        om1: n1.mix.HumidityRatio.Hint,
        beta0: n0.mix.Wetness.Hint,
        beta1: n1.mix.Wetness.Hint,
    ):
        return om0 * (1 - beta1) - om1 * (1 - beta0)


class PhaseEquilibrium(EquationBase):
    """
    Thermodynamic equilibrium between vapour and liquid at the static state
    of a node: the vapour is saturated, ``phi = p_v / p_sat(T) = 1``.
    """

    def residual(self, phi0: n0.stc.RelHumidity.Hint):
        return phi0 - 1


class ConstantAreaMomentum(EquationBase):
    """
    Streamwise momentum balance between two stations of a frictionless duct
    of constant area: ``p0 + rho0 Vm0^2 = p1 + rho1 Vm1^2``.
    """

    def residual(
        self,
        p0: n0.stc.Pressure.Hint,
        rho0: n0.stc.Density.Hint,
        vm0: n0.kin.V_mer.Hint,
        p1: n1.stc.Pressure.Hint,
        rho1: n1.stc.Density.Hint,
        vm1: n1.kin.V_mer.Hint,
    ):
        return (p0 + rho0 * vm0**2) - (p1 + rho1 * vm1**2)


class EquilibriumMachNumber(EquationBase):
    """
    Mach number based on the equilibrium (two-phase) speed of sound,
    ``M_e = V / a_e``. Only meaningful at saturated nodes.
    """

    def residual(
        self,
        mach_eq0: n0.mix.EqMach.Hint,
        v0: n0.kin.V_mag.Hint,
        a_eq0: n0.stc.EqSpeedSound.Hint,
    ):
        return mach_eq0 * a_eq0 - v0
