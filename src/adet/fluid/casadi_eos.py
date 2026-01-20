import logging
from typing import Any, Callable, ClassVar, Generic, TypeVar, cast, overload
import casadi as cs
import CoolProp as cp
import jax
import scipy.differentiate as diff

from adet.constants import COOLPROP_NAMES_MAP
from adet.fluid.settings import ExternalFluidModel
from adet.tools.coolprop_utils import DebugAbstractState, get_input_names


logger = logging.getLogger(__name__)

# This is needed to keep references alive
_JAC_CALLBACK_CACHE = []
_HES_CALLBACK_CACHE = []
_VAC_CALLBACK_CACHE = []

# Properties whose first derivative does not exist
NOT_JACOBIABLE = ['viscosity']
# Properties whose second derivative does not exist
NOT_HESSIABLE = ['speed_sound', 'cpmass', 'cvmass']
# NOTE: Where derivatives are not available, we use 0.0
# an alternative would be to code finite differences
# for spe


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

    @staticmethod
    def diag(*args):
        return cs.Sparsity.diag(*args)


# *** DUMMY HELPER FUNCTIONS FOR DEV
# These are needed to mock the actual function,
# and get information about the shapes required
# when overloading the methods
# ( 2 inputs ) |-> ( 4 outputs ) example
def dummy_eos(v0, v1, n_out):
    return n_out * [cs.sin(v0) + cs.cos(v1)]


def get_dummy_jac_shape(n_out: int, n_span: int, debug=False) -> cs.Function:
    v0 = MX.sym('v0', n_span)
    v1 = MX.sym('v1', n_span)
    dummy_expr = dummy_eos(v0, v1, n_out)

    dummy_func = cs.Function('dummy', [v0, v1], dummy_expr)
    if debug:
        print(f'|> [DEV NOTE] Needed jacobian shape:\n\t{dummy_func.jacobian()}\n')

    return dummy_func.jacobian()


def get_dummy_hess_shape(n_out: int, n_span: int, debug=False) -> cs.Function:
    dummy_jac = get_dummy_jac_shape(n_out, n_span)
    if debug:
        print(f'|> [DEV NOTE] Needed hessian shape:\n\t{dummy_jac.jacobian()}\n')
    return dummy_jac.jacobian()


def get_dummy_vacc_shape(n_out: int, n_span: int, debug=False) -> cs.Function:
    dummy_hess = get_dummy_hess_shape(n_out, n_span)
    if debug:
        print(f'|> [DEV NOTE] Needed vaccarian shape:\n\t{dummy_hess.jacobian()}\n')
    return dummy_hess.jacobian()


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
        name: str,
        eos: Any,
        input_pair: int,
        output_props: list[str] | tuple[str, ...],
        num_span: int = 1,
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

    def has_jacobian(self, *args) -> bool:
        return True

    def get_jacobian(self, name, inames, onames, opts={}):
        self.hes_callback = CasadiEosHessian(
            name=f'{name}',
            eos=self._eos,
            input_pair=self._input_pair,
            output_props=self._output_props,
            num_span=self._num_span,
            opts=opts,
        )

        _HES_CALLBACK_CACHE.append(self.hes_callback)

        return self.hes_callback

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

                    if prop in NOT_JACOBIABLE:
                        derivative = fwd_diff(
                            eos,
                            input_pair,
                            prop,
                            updt_vals[0],
                            updt_vals[1],
                            input_idx,
                        )
                        # derivative = 0.0
                    else:
                        derivative = eos.first_partial_deriv(
                            prop_id, input_id, other_id
                        )

                    result[prop_idx][input_idx][span, span] = derivative

        return jax.tree.leaves(result)


def fwd_diff(eos, input_pair: int, prop, x, y, wrt: int, eps: float = 1e-4):
    """Simple forward difference for an AbstractState"""
    prop_meth = getattr(eos, prop)
    prop_orig = prop_meth()
    match wrt:
        case 0:
            eps *= x
            x_pert = x + eps
            eos.update(input_pair, x_pert, y)
            prop_pert = prop_meth()
        case 1:
            eps *= y
            y_pert = y + eps
            eos.update(input_pair, x, y_pert)
            prop_pert = prop_meth()

    # NOTE: In theory I should restore the eos to its original state
    # this is actually slightly modifying the jacobian,
    # but saves a lot of updates and impact is minimal
    # >>> eos.update(input_pair, x, y)

    return (prop_pert - prop_orig) / eps


class CasadiEosHessian(cs.Callback):
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
        logger.debug(f'Hessian reference {self.name()} deleted')

    def get_n_in(self):
        return (
            len(self._input_names)
            + len(self._output_props)
            + len(self._input_names) * len(self._output_props)
        )

    def get_n_out(self):
        return (
            len(self._input_names)
            * len(self._output_props)
            * (len(self._output_props) + len(self._input_names))
        )

    def get_sparsity_in(self, i):
        ins_and_outs = len(self._input_names) + len(self._output_props)

        if i < ins_and_outs:
            return Sparsity.dense(self._num_span)
        else:
            return Sparsity.diag(self._num_span)

    def get_sparsity_out(self, i):
        ins_and_outs = len(self._input_names) + len(self._output_props)

        # WARN: This if loop assumes two input properties
        if i % ins_and_outs == 0 or i % ins_and_outs == 1:
            pattern = Sparsity(self._num_span**2, self._num_span)
            for sp in range(self._num_span):
                pattern.add_nz((self._num_span + 1) * sp, sp)
            return pattern
        else:
            return Sparsity(self._num_span**2, self._num_span)

    def has_jacobian(self, *args) -> bool:
        return True

    def get_jacobian(self, name, inames, onames, opts={}):
        self.vac_callback = CasadiEosVaccarian(
            name=f'{name}',
            output_props=self._output_props,
            num_span=self._num_span,
            opts=opts,
        )

        _VAC_CALLBACK_CACHE.append(self.vac_callback)

        return self.vac_callback

    def eval(self, args):
        eos = self._eos
        num_span = self._num_span
        input_names = self._input_names
        output_props = self._output_props
        input_pair = self._input_pair

        ins_and_outs = len(self._input_names) + len(self._output_props)

        result = [
            [
                [cs.DM(num_span**2, num_span) for _ in range(ins_and_outs)]
                for _ in input_names
            ]
            for _ in output_props
        ]

        for span_idx in range(num_span):
            updt_vals = [float(args[i][span_idx]) for i in range(len(input_names))]
            eos.update(input_pair, *updt_vals)

            for prop_idx, prop in enumerate(output_props):
                prop_name = COOLPROP_NAMES_MAP.get(prop, prop)
                # Get the integer id of that property
                prop_id = getattr(cp, f'i{prop_name}')

                for in0_idx, inpt0 in enumerate(input_names):
                    for in1_idx, inpt1 in enumerate(input_names):
                        # Get the input properties ids
                        inp0_id = getattr(cp, f'i{inpt0}')
                        oth0_id = getattr(cp, f'i{input_names[1 - in0_idx]}', None)

                        inp1_id = getattr(cp, f'i{inpt1}')
                        oth1_id = getattr(cp, f'i{input_names[1 - in1_idx]}', None)

                        if prop in (NOT_JACOBIABLE + NOT_HESSIABLE):
                            derivative = 0.0
                        else:
                            derivative = eos.second_partial_deriv(
                                prop_id, inp0_id, oth0_id, inp1_id, oth1_id
                            )

                        result[prop_idx][in0_idx][in1_idx][
                            (num_span + 1) * span_idx, span_idx
                        ] = derivative

        return jax.tree.leaves(result)


class CasadiEosVaccarian(cs.Callback):
    """Third order partial derivative tensor, this just returns zeros"""

    def __init__(
        self,
        name,
        output_props,
        num_span,
        opts=None,
    ):
        super().__init__()
        # Assignments
        self._num_span = num_span
        self.dummy_func = get_dummy_vacc_shape(len(output_props), num_span, False)
        self.construct(name, opts or {})

    def __del__(self):
        logger.debug(f'Third order deriv tensor reference {self.name()} deleted')

    def get_n_in(self):
        return self.dummy_func.n_in()

    def get_n_out(self):
        return self.dummy_func.n_out()

    def get_sparsity_in(self, i):
        return self.dummy_func.sparsity_in(i)

    def get_sparsity_out(self, i):
        return self.dummy_func.sparsity_out(i)

    def has_jacobian(self, *args) -> bool:
        return False

    def eval(self, args):
        return [cs.DM.zeros(self.get_sparsity_out(i)) for i in range(self.get_n_out())]


M = TypeVar('M', bound=ExternalFluidModel)


class CasadiEosFactory(Generic[M]):
    instance_counter: ClassVar[int] = 0

    def __init__(self, fluid_model: M) -> None:
        self.fluid_model = fluid_model

    def make_eos(
        self,
        input_pair: int,
        output_quantities: tuple[str, ...] | list[str],
        length: int,
        name: str = '',
    ):
        pair_name = ''.join(get_input_names(input_pair))

        if not name:
            name = f'eos_{pair_name}_l{length}'

        eos_callback = CasadiEos(
            name,
            self.fluid_model.eos_object,
            input_pair,
            output_quantities,
            length,
        )

        return eos_callback


# ------------------- USAGE EXAMPLE -------------------

if __name__ == '__main__':
    from numpy.typing import ArrayLike

    # Setup a CoolProp EOS
    eos = DebugAbstractState('HEOS', 'Air')

    NUM_SPAN = 7
    PROPERTIES = [
        'hmass',
        'smass',
        'rhomass',
        'speed_sound',
        'viscosity',
    ]
    OPTS = {
        'enable_fd': True,
        'fd_method': 'forward',
    }

    # Example
    callback = CasadiEos(
        'PT_eos',
        eos,
        cp.PT_INPUTS,
        PROPERTIES,
        NUM_SPAN,
        opts=OPTS,
    )

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

    hes_value = callback.jacobian().jacobian()(
        v0_val, v1_val, *callback(v0_val, v1_val), *jac_value
    )

    print(f'Hessian is:\n{hes_value}')

    vac_value = (
        callback.jacobian()
        .jacobian()
        .jacobian()(v0_val, v1_val, *callback(v0_val, v1_val), *jac_value, *hes_value)
    )
