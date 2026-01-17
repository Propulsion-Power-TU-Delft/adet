from typing import Literal

import casadi as cs
import numpy as np

from sympy import Symbol
from pint.facets.plain import PlainQuantity

from adet.fluid.casadi_eos import CasadiEos


def safe_abs(x):
    return (x**2) ** 0.5


def safe_sum(x):
    return (x**0).T @ x


def safe_mean(x):
    size = max(x.shape)
    return safe_sum(x) / size


def safe_min_clip(x, min_value):
    """
    Lower clipping of the absolute vaue of x
    with respect to a minimum value.
    Type safe for casadi, numpy and pint
    """
    if is_casadi_type(x):
        # NOTE: If x is exactly 0 the normal sign function
        # is problematic
        x_sign = cs.if_else(x >= 0, 1, -1)
        x = x_sign * cs.fmax(cs.fabs(x), min_value)
    elif isinstance(x, PlainQuantity):
        x_sign = np.sign(x.magnitude)
        x = x_sign * np.clip(np.abs(x.magnitude), min_value, None) * x.units
    elif isinstance(x, Symbol):
        pass
    else:
        x_sign = np.sign(x)
        x = x_sign * np.clip(np.abs(x), min_value, None)

    return x


def therodynamic_derivative(eos: CasadiEos, arg0, arg1, wrt: Literal[0, 1]):
    eos_value = eos(arg0, arg1)
    if not isinstance(eos_value, tuple):
        eos_value = (eos_value,)

    jacobian = eos.jacobian()(arg0, arg1, *eos_value)
    return [cs.diag(jacobian[wrt + 2 * i]) for i in range(eos.n_out())]


def trapezoid1(y, x):
    """Trapezoidal rule"""
    dx = x[1:, :] - x[:-1, :]
    integrand = (y[:-1, :] + y[1:, :]) * dx / 2
    return cs.sum1(integrand)


def trapezoid2(y, x):
    """Trapezoidal rule"""
    dx = x[:, 1:] - x[:, :-1]
    integrand = (y[:, :-1] + y[:, 1:]) * dx / 2
    return cs.sum2(integrand)


def is_casadi_type(x):
    return isinstance(x, (cs.DM, cs.MX, cs.SX))


def fin_diff(f, x, edge_order: Literal['first', 'second'] = 'second'):
    """
    Centered finite difference df/dx derivative along the span,
    fwd and bwd on the edges
    """
    first_delta_x = safe_min_clip(x[1] - x[0], 1e-8)
    inner_delta_x = safe_min_clip(x[2:] - x[:-2], 1e-8)
    final_delta_x = safe_min_clip(x[-1] - x[-2], 1e-8)

    if edge_order == 'first':
        first_der = (f[1] - f[0]) / first_delta_x
        final_der = (f[-1] - f[-2]) / final_delta_x
    else:
        first_der = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * first_delta_x)
        final_der = (f[-3] - 4 * f[-2] + 3 * f[-1]) / (2 * final_delta_x)

    internal_der = (f[2:] - f[:-2]) / inner_delta_x

    return cs.vertcat(first_der, internal_der, final_der)
