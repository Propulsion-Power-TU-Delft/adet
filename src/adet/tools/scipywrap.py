import numpy as np
from numpy import inf


class ScipyObjectiveFunctionWrapper:
    """
    Wrapper class for scipy minimize methods to stop solver when a certain value of
    the objective function is reached
    """

    def __init__(self, fun, fun_tol=None, max_it=20, verbose=0):
        self.fun = fun
        self.best_x = None
        self.best_f = inf
        self.fun_tol = fun_tol or -inf
        self.number_of_f_evals = 0
        self.max_it = max_it
        self.verbose = verbose

    def __call__(self, x, *args):
        _f = self.fun(x, args)

        self.number_of_f_evals += 1

        if self.verbose == 2:
            print(' %d\t\t\t%f ' % (self.number_of_f_evals, _f))

        if _f < self.best_f:
            self.best_x, self.best_f = x, _f

        return _f

    def stop(self, *args):
        if self.best_f < self.fun_tol or self.number_of_f_evals == self.max_it:
            raise Trigger


class Trigger(Exception):
    pass


def roots_2(a, b, c):
    return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a), (
        -b - np.sqrt(b**2 - 4 * a * c)
    ) / (2 * a)
