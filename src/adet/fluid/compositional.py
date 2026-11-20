"""
Interface for fluids whose thermodynamic state depends on node *composition*
variables in addition to the two-property update pair (e.g. humid air, whose
state needs the humidity ratio and the wetness).

A compositional fluid is used exactly like a pure fluid in
:class:`adet.fluid.settings.FluidSettings`. The assembler then

- adds the composition variables at every node as system arguments (free
  unknowns, unless fixed by boundary conditions), shared by the static,
  total and relative-total states of the node (frozen composition between
  states);
- passes the node composition to every node-level EOS evaluation and, for
  equation-level EOS calls (``self.eos(...)`` inside a residual), the
  composition of the *first* node the equation is placed on.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import casadi as cs

from adet.varspec import VarSpec


class CompositionalFluid(ABC):
    @property
    @abstractmethod
    def composition_variables(self) -> tuple[VarSpec, ...]:
        """Node-level (stateless) composition variables, as global specs."""

    @abstractmethod
    def make_eos(
        self,
        input_pair: int,
        out_properties: Sequence[str],
        length: int,
        name: str,
    ) -> cs.Function:
        """
        CasADi function ``(v0, v1, *composition) -> outputs``, all column
        vectors of size ``length``. ``input_pair`` is a CoolProp input pair id,
        ``out_properties`` are property names as in ``VarSpec.symbol``.
        """
