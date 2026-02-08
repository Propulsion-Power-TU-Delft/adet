from inspect import getfullargspec
from typing import Any, Callable
import casadi as cs

import CoolProp as cp
from numpy.typing import NDArray
import sympy as sm

from adet.tools.coolprop_utils import pair_tuple_from_id


class IdealGasEos:
    solution_cache: dict[int, dict[str, Callable]] = {}

    def __init__(self, gamma, gas_constant):
        self.current_state: dict[str, NDArray] = {}
        self._gamma = gamma
        self._gas_constant = gas_constant

        self._cvmass = self._gas_constant / (self._gamma - 1)
        self._cpmass = self._cvmass * self._gamma

    def eos(self, p, T, rhomass, hmass, smass, speed_sound):
        r1 = p / rhomass - self._gas_constant * T
        r2 = hmass - self._cpmass * T
        r3 = smass - self._cvmass * T
        r4 = speed_sound - (self._gamma * self._gas_constant * T) ** 0.5

        return r1, r2, r3, r4

    @property
    def arguments(self):
        return getfullargspec(self.eos).args[1:]

    def update(self, input_pair, value0, value1):
        input_vars = pair_tuple_from_id(input_pair)
        other_vars = set(self.arguments).difference(input_vars)

        function_inputs = {
            input_vars[0]: value0,
            input_vars[1]: value1,
        }

        cache_hit = input_pair in self.solution_cache

        if cache_hit:
            solution_funcs = self.__class__.solution_cache[input_pair]
        else:
            symbols = {arg: sm.Symbol(arg) for arg in self.arguments}
            symbolic_func = self.eos(**symbols)
            symbolic_solution = sm.solve(symbolic_func, other_vars)

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

    def hmass(self):
        return self.current_state['hmass']

    def smass(self):
        return self.current_state['smass']

    def cpmass(self):
        return self._cpmass

    def cvmass(self):
        return self._cvmass

    def speed_sound(self):
        return self.current_state['speed_sound']


if __name__ == '__main__':
    eos = IdealGasEos(1.4, 287.0)
    eos.update(cp.PT_INPUTS, 1e5, 300)
