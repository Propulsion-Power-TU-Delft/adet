from abc import ABC, abstractmethod
from inspect import getfullargspec
import logging
from typing import Any, Callable

import CoolProp as cp
import sympy as sm

from adet.constants import COOLPROP_PAIRS
from adet.tools.coolprop_utils import pair_tuple_from_id

logger = logging.getLogger(__name__)


class SymbolicAbstractState(ABC):
    solution_cache: dict[int, dict[str, Callable]] = {}

    def __init__(self, gamma, gas_constant, viscosity=1e-5):
        # TODO: Add extra optional manual properties input, e.g. viscosity
        # Move gamma and gas_constant to subclasses, make this general
        self.current_state: dict[str, Any] = {}
        self._gamma = gamma
        self._gas_constant = gas_constant

        self._viscosity = viscosity
        self._cvmass = self._gas_constant / (self._gamma - 1)
        self._cpmass = self._cvmass * self._gamma

    @abstractmethod
    def eos(self, *args):
        raise NotImplementedError

    @property
    def arguments(self):
        return getfullargspec(self.eos).args[1:]

    def update(self, input_pair, value0, value1):
        input_vars = pair_tuple_from_id(input_pair)
        other_vars = set(self.arguments).difference(input_vars)
        logger.debug('Updating {self} with {input_vars}')

        function_inputs = {
            input_vars[0]: value0,
            input_vars[1]: value1,
        }

        cache_hit = input_pair in self.solution_cache

        if cache_hit:
            logger.debug(f'Cache hit for {input_vars}')
            solution_funcs = self.__class__.solution_cache[input_pair]
        else:
            logger.debug(f'Cache miss for {input_vars}, building symbolic solution')
            symbols = {arg: sm.Symbol(arg) for arg in self.arguments}
            symbolic_func = self.eos(**symbols)
            symbolic_solution = sm.solve(symbolic_func, other_vars)

            if isinstance(symbolic_solution, list):
                symbolic_solution = symbolic_solution[0]

            for name in input_vars:
                symbol = symbols[name]
                symbolic_solution[symbol] = symbol

            solution_funcs = {
                symbol.name: sm.lambdify(input_vars, expr)
                for symbol, expr in symbolic_solution.items()
            }

        self.current_state = {
            sym: func(**function_inputs) for sym, func in solution_funcs.items()
        }

        self.__class__.solution_cache[input_pair] = solution_funcs

    def p(self):
        return self.current_state['p']

    def T(self):
        return self.current_state['T']

    def rhomass(self):
        return self.current_state['rhomass']

    def hmass(self):
        return self.current_state['hmass']

    def smass(self):
        return self.current_state['smass']

    def cpmass(self):
        return self._cpmass

    def cvmass(self):
        return self._cvmass

    def viscosity(self):
        return self._viscosity

    def speed_sound(self):
        return self.current_state['speed_sound']


class IdealGasState(SymbolicAbstractState):
    def eos(self, p, T, rhomass, hmass, umass, smass, speed_sound):
        r1 = p - self._gas_constant * rhomass * T
        r2 = hmass - self._cpmass * T
        r3 = umass - self._cvmass * T
        r4 = speed_sound - (self._gamma * self._gas_constant * T) ** 0.5
        r5 = smass - self._cpmass * sm.log(T) + self._gas_constant * sm.log(p)

        return r1, r2, r3, r4, r5


if __name__ == '__main__':
    import casadi as cs

    eos = IdealGasState(
        gamma=1.4,
        gas_constant=287.0,
    )

    # Polymorphic
    eos.update(
        cp.PT_INPUTS,
        1e5,
        300,
    )

    for pair in COOLPROP_PAIRS.keys():
        if 'Q' in COOLPROP_PAIRS[pair]:
            continue

        print(f'Update {COOLPROP_PAIRS[pair]}')

        eos.update(
            pair,
            cs.MX.sym('x', 5),  # pyright: ignore
            cs.MX.sym('y', 5),  # pyright: ignore
        )
