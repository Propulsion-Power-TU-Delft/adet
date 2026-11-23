import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import getfullargspec
from typing import Any, Callable, ClassVar, Sequence, cast

import casadi as cs
import numpy as np
from pint import Unit
from pint.facets.plain import PlainQuantity
from pint.registry import Quantity

from adet.fluid.casadi_eos import CasadiEos
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, VarSpec

logger = logging.getLogger(__name__)


class EmbeddedEos:
    """
    View of an equation's EoS, built on access by ``EquationBase.eos``.

    Output properties come from the equation config; the callable is the one
    bound by the assembler at build time (``None`` before that). Called with
    Pint quantities (unit checks) it passes the output units through,
    otherwise it forwards to the bound callable.
    """

    def __init__(
        self,
        parent: 'EquationBase',
        out_pties: Sequence[VarSpec],
        callable: CasadiEos | cs.Function | Any | None = None,
    ):
        self.parent = parent
        self.out_pties = out_pties
        self.callable = callable

    def _pass_units(self, *args) -> PlainQuantity | list[PlainQuantity]:
        lengths = {len(a) for a in args}
        if len(lengths) > 1:
            raise ValueError('Incompatible thermodynamic lengths, strange...')

        num_span = lengths.pop()

        if not self.out_pties:
            raise AttributeError(
                f'{self.parent} is calling an EoS without output properties'
            )

        all_outputs = [
            Quantity(num_span * [np.nan], spec.unit) for spec in self.out_pties
        ]

        if len(all_outputs) == 1:
            return all_outputs[0]
        else:
            return all_outputs

    def __call__(self, *args) -> Any | Sequence[Any]:
        is_quantity = [isinstance(q, PlainQuantity) for q in args]

        if any(is_quantity):
            return self._pass_units(*args)
        elif self.callable is None:
            raise AttributeError(
                f'Equation of state for {self.parent} was not assignned at runtime'
            )
        else:
            return self.callable(*args)


@dataclass(frozen=True)
class EquationConfig:
    """Configuration for equation class properties"""

    input_pair: int = 0
    manual_units: tuple[str, ...] = ()
    scaling_factor: tuple[float | None, ...] | None = None
    out_properties: tuple[VarSpec, ...] = ()


class EquationBase(ABC):
    """
    Base Class for defining equations, including argument validation and organization,
    node variable creation and simple storage of the last arguments.

    Supports argument aliasing: allowing the residual function signature to use
    different names than the system-level variable names.
    """

    config: ClassVar[EquationConfig] = EquationConfig()

    def __init__(self, custom_scaling_factor: list[float] | None = None):
        """
        Parameters
        ----------
        custom_scaling_factor : list[float] | None
            Custom scaling factors for equations
        """

        self._arguments: tuple[str, ...] = ()

        if custom_scaling_factor:
            self._scaling_factor = custom_scaling_factor
        else:
            self._scaling_factor = self.config.scaling_factor

    @abstractmethod
    def residual(self, *args) -> Any | tuple[Any, ...]: ...

    @property
    def arg_symbols(self):
        if not self._arguments:
            arguments = []
            var_specs = self._get_args_specs()
            for var_spec in var_specs:
                if var_spec.state is not None:
                    state = str(var_spec.state.value)
                else:
                    state = ''

                arguments.append(state + var_spec.symbol + str(var_spec.node))
            self._arguments = tuple(arguments)

        return self._arguments

    @property
    def arg_units(self):
        vars_specs = self._get_args_specs()
        return [Unit(s.unit) for s in vars_specs]

    @property
    def num_args(self):
        return len(self._arguments)

    @property
    def arg_names(self) -> list[str]:
        return getfullargspec(self.residual).args[1:]

    @property
    def arg_specs(self) -> list[VarSpec]:
        return self._get_args_specs()

    @property
    def arg_nodes(self) -> list[int]:
        vars_specs = self._get_args_specs()
        return sorted({s.node for s in vars_specs})

    def _get_args_specs(self) -> list[VarSpec]:
        residual_specs = getfullargspec(self.residual)
        all_args = residual_specs.args[1:]
        args_hints = residual_specs.annotations

        vars_specs = []
        for arg in all_args:
            # NOTE: Only use the first annotation is used by convention!
            hint = args_hints[arg]
            spec = cast(VarSpec, hint.__metadata__[0])
            logger.debug(f'Variable is {spec}')

            vars_specs.append(spec)

        self._check_duplicates(vars_specs)

        return vars_specs

    def _check_duplicates(self, variables_specs: list[VarSpec]):
        # Check for duplicate arguments
        seen = set()
        duplicates = set()
        for sp in variables_specs:
            if sp in seen:
                duplicates.add(sp)
            else:
                seen.add(sp)

        if duplicates:
            raise ValueError(
                f'Duplicate argument(s) in {self.__class__.__name__}, {duplicates}'
            )

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, 'config'):
            cls.config = EquationConfig()

        config = cls.config
        # This checks if both exist
        if bool(config.out_properties) != bool(config.input_pair):
            raise ValueError(
                f'Please specify both input_pair and out_properties in {cls}'
            )

        if config.manual_units:
            logger.debug(f'{cls.__name__} defines its units manually')

        return super().__init_subclass__()

    @property
    def eos(self) -> EmbeddedEos:
        # Built on access from the config and the callable the assembler bound to this
        # instance at build time, if any
        if not self.config.input_pair:
            raise AttributeError(
                f'{self.__class__.__name__} does not declare an EoS in its config'
            )
        return EmbeddedEos(
            self,
            [pty.Glob for pty in self.config.out_properties],
            self.__dict__.get('_bound_eos'),
        )

    @eos.setter
    def eos(self, eos: cs.Function | CasadiEos | Callable):
        # Per instance: with compositional fluids the EoS is bound to the
        # composition of the nodes each instance is placed on
        if not self.config.input_pair:
            raise AttributeError(
                f'{self.__class__.__name__} does not declare an EoS in its config'
            )
        logger.debug(f'Assigning EoS for {self.__class__} instance')
        self._bound_eos = eos

    def __getstate__(self):
        # Runtime EoS (CasADi objects) is rebuilt by the assembler on build()
        state = self.__dict__.copy()
        state.pop('_bound_eos', None)
        return state


class UniqueEquation(EquationBase):
    """
    Inherit this for all equations families which can be defined only
    once per component, either on one or two of the component nodes
    """

    def __init__(self, scaling_factor: list[float] | None = None):
        if self.__class__.__base__ == UniqueEquation:
            raise TypeError(f'Do not inherit directly from {self}')
        super().__init__(scaling_factor)


# fmt: off
class LossApplier(UniqueEquation): ...
class VelocityTriangles(UniqueEquation): ...
class DeviationModel(UniqueEquation): ...
class IncidenceModel(UniqueEquation): ...
class CamberLineGeom(UniqueEquation): ...
class MeridionalGeom(UniqueEquation): ...
class MeridAreaBlockage(UniqueEquation): ...
class CompositionModel(UniqueEquation): ...
# fmt: on


if __name__ == '__main__':
    setup_logger(logger, logging.INFO, logging.INFO)
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

    eq = DummyEq()
    print(eq.arg_symbols)
