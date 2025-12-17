from itertools import chain
import logging
from typing import Any, Callable, ClassVar, Generic, TypeVar, cast, overload
import casadi as cs
import CoolProp as cp

from adet.constants import COOLPROP_NAMES_MAP
from adet.fluid.settings import ExternalFluidModel
from adet.tools.coolprop_utils import (
    DebugAbstractState,
    get_input_names,
    pair_id_from_name,
    pair_id_from_tuple,
    pair_name_from_tuple,
)


logger = logging.getLogger(__name__)

# This is needed to keep references alive
_JAC_CALLBACK_CACHE = []


# These two classes are just to correct meaningless
# type warnings (Some stubs are wrong in CasADi's python API)
class MX(cs.MX):
    @staticmethod
    def sym(*args) -> cs.MX:
        return cs.MX.sym(*args)


class SX(cs.SX):
    @staticmethod
    def sym(*args) -> cs.SX:
        return cs.SX.sym(*args)


class DM(cs.DM):
    @staticmethod
    def ones(*args) -> cs.DM:
        return cs.DM.ones(*args)


class Sparsity(cs.Sparsity):
    @staticmethod
    def dense(*args):
        return cs.Sparsity.dense(*args)


# *** DUMMY HELPER FUNCTIONS FOR DEV
# These are needed to mock the actual function,
# and get information about the shapes required
# when overloading the methods
# ( 2 inputs ) |-> ( 4 outputs ) example
def dummy_eos(v0, v1):
    return (
        cs.sin(v0 + v1),
        cs.cos(v0 + v1),
        cs.sin(v1 - v0),
        cs.cos(v1 - v0),
    )


def get_dummy_jac_shape(v0: cs.MX, v1: cs.MX):
    dummy_expr = dummy_eos(v0, v1)
    dummy_func = cs.Function('dummy', [v0, v1], dummy_expr)
    print(f'|> [DEV NOTE] Needed jacobian shape:\n\t{dummy_func.jacobian()}\n')

    return dummy_func.jacobian()


class CasadiEos(cs.Callback):
    def __init__(
        self,
        name: str,
        eos: Any,
        input_pair: int,
        output_props: list[str] | tuple[str, ...],
        num_span: int = 1,
        opts={},
    ):
        # Assignments
        cs.Callback.__init__(self)
        self._eos = eos
        self._input_pair = input_pair
        self._output_props = output_props
        self._num_span = num_span

        # Post
        self._input_names = get_input_names(input_pair)
        self.construct(name, opts)

    def __del__(self):
        logger.debug(f'Callback reference {self.name()} deleted')

    def __deepcopy__(self, dummy=None):
        return self.__class__(
            self.name(),
            self._eos,
            self._input_pair,
            self._output_props,
            self._num_span,
        )

    def get_n_in(self):
        return len(self._input_names)

    def get_n_out(self):
        return len(self._output_props)

    def get_sparsity_in(self, i):
        return Sparsity.dense(self._num_span)

    def get_sparsity_out(self, i):
        return Sparsity.dense(self._num_span)

    def get_name_in(self, i):
        return self._input_names[i]

    def get_name_out(self, i):
        return self._output_props[i]

    def eval(self, args):
        num_span = self._num_span
        results = [cs.DM(self.get_sparsity_out(i)) for i in self._output_props]

        for span in range(num_span):
            updt_vals = [float(args[i][span]) for i, _ in enumerate(self._input_names)]

            self._eos.update(self._input_pair, *updt_vals)

            for j, prop in enumerate(self._output_props):
                # Get the property method
                prop_meth = getattr(self._eos, prop)

                # Add to the results
                results[j][span] = prop_meth()

        return results

    @overload
    def __call__(self, var0: cs.MX, var1: cs.MX) -> tuple[cs.MX, ...] | cs.MX: ...

    @overload
    def __call__(self, var0: cs.DM, var1: cs.DM) -> tuple[cs.DM, ...] | cs.DM: ...

    def __call__(self, var0: Any, var1: Any) -> tuple[Any, ...] | Any:
        return super().__call__(var0, var1)

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts={}):
        self.jac_callback = CasadiEosJacobian(
            name=f'{name}',
            eos=self._eos,
            input_pair=self._input_pair,
            output_props=self._output_props,
            num_span=self._num_span,
            opts=opts,
        )

        _JAC_CALLBACK_CACHE.append(self.jac_callback)

        return self.jac_callback


class CasadiEosJacobian(cs.Callback):
    def __init__(
        self,
        name,
        eos,
        input_pair,
        output_props,
        num_span,
        opts=None,
    ):
        super().__init__()
        # Assignments
        self._eos = eos
        self._input_pair = input_pair
        self._output_props = output_props
        self._num_span = num_span

        # Post
        self._input_names = get_input_names(input_pair)
        self.construct(name, opts or {})

    def __del__(self):
        logger.debug(f'Jacobian reference {self.name()} deleted')

    def get_n_in(self):
        return len(self._input_names) + len(self._output_props)

    def get_n_out(self):
        return len(self._input_names) * len(self._output_props)

    def get_sparsity_in(self, i):
        return Sparsity.dense(self._num_span)

    def get_sparsity_out(self, i):
        return Sparsity.diag(self._num_span)

    def eval(self, args):
        eos = self._eos
        num_span = self._num_span
        input_names = self._input_names
        output_props = self._output_props
        input_pair = self._input_pair

        result = [
            [cs.DM(num_span, num_span) for _ in input_names] for _ in output_props
        ]

        for span in range(num_span):
            updt_vals = [float(args[i][span]) for i in range(len(input_names))]
            eos.update(input_pair, *updt_vals)

            for input_idx, inpt in enumerate(input_names):
                for prop_idx, prop in enumerate(output_props):
                    # If name is not in the map, just keep
                    # its original name, to which we add a `i`
                    # e.g. rhomass -> iDmass, but speed_sound -> ispeed_sound
                    prop_name = COOLPROP_NAMES_MAP.get(prop, prop)

                    # Get the integer id of that property
                    prop_id = getattr(cp, f'i{prop_name}')

                    # Get the input properties ids
                    input_id = getattr(cp, f'i{inpt}')
                    other_id = getattr(cp, f'i{input_names[1 - input_idx]}', None)

                    # Only work for couples (pairs)
                    if other_id is None:
                        raise NotImplementedError('Only pairs supported')

                    result[prop_idx][input_idx][span, span] = eos.first_partial_deriv(
                        prop_id, input_id, other_id
                    )

        return list(chain.from_iterable(result))


M = TypeVar('M', bound=ExternalFluidModel)


class CasadiEosFactory(Generic[M]):
    instance_counter: ClassVar[int] = 0

    def __init__(self, fluid_model: M) -> None:
        self.fluid_model = fluid_model
        self.__class__.instance_counter += 1

    def make_eos(
        self,
        input_quantities: tuple[str, ...] | list[str],
        output_quantities: tuple[str, ...] | list[str],
        length: int,
    ):
        pair_name = pair_name_from_tuple(tuple(input_quantities))
        pair_id = pair_id_from_name(pair_name)

        eos_callback = CasadiEos(
            f'eos_{pair_name}_n{self.instance_counter}_l{length}',
            self.fluid_model.eos_object,
            pair_id,
            output_quantities,
            length,
        )

        return eos_callback


# ------------------- USAGE EXAMPLE -------------------

if __name__ == '__main__':
    from numpy.typing import ArrayLike

    # Setup a CoolProp EOS
    eos = DebugAbstractState('HEOS', 'Air')

    NUM_SPAN = 10
    PROPERTIES = [
        'hmass',
        'smass',
        'speed_sound',
    ]
    OPTS = {'enable_fd': True}

    # Example
    callback = CasadiEos('PT_eos', eos, cp.PT_INPUTS, PROPERTIES, NUM_SPAN)

    # Type annotation
    callback = cast(
        Callable[
            [ArrayLike, ArrayLike],
            tuple[cs.DM, ...],
        ],
        callback,
    )

    v0_val = cs.linspace(100e5, 600e5, NUM_SPAN)  # Pressure [Pa]
    v1_val = cs.linspace(400, 600, NUM_SPAN)  # Temperature [K]

    value_str = '\n\t'.join(
        list(
            map(lambda x: x.__str__(), callback(v0_val, v1_val)),
        )
    )

    print(f'Function value:\n\t{value_str}')

    #  *** ACUTAL JACOBIAN ***
    jac_value = callback.jacobian()(
        v0_val,
        v1_val,
        *callback(v0_val, v1_val),
    )

    print(
        f'Jacobian value of properties {PROPERTIES} '
        f'wrt {callback._input_names} input pair:\n{jac_value}'
    )
