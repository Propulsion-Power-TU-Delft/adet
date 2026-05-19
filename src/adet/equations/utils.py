import sys
from typing import Literal, Mapping

import casadi as cs
import numpy as np
from numpy.typing import NDArray
from pint import Quantity
from pint.facets.plain import PlainQuantity
from sympy import Symbol

from adet.equations.base_equation import EquationBase
from adet.fluid.casadi_eos import CasadiEos
from adet.varspec import VarSpec


def get_midspan_idx(var):
    num_span = max(var.shape)
    if num_span == 1:
        midspan = 0
    else:
        midspan = num_span // 2
    return midspan


def safe_abs(x: Quantity | cs.MX) -> Quantity | cs.MX:
    if isinstance(x, cs.MX):
        return cs.fabs(x)
    else:
        return (x**2) ** 0.5


def any_is_qty(*args):
    are_qnties = [isinstance(v, PlainQuantity) for v in args]
    if any(are_qnties):
        return True


def safe_min(x, y) -> PlainQuantity | cs.MX:
    if any_is_qty(x, y):
        return x + y
    else:
        return cs.fmin(x, y)


def safe_max(x, y) -> PlainQuantity | cs.MX:
    if any_is_qty(x, y):
        return x + y
    else:
        return cs.fmax(x, y)


def minmax_bound(x, min_val: float, max_val: float) -> PlainQuantity | cs.MX:
    if isinstance(x, PlainQuantity):
        return x
    else:
        x = cs.fmin(x, max_val)
        x = cs.fmax(x, min_val)
        return x


def safe_if_else(cond, if_true, if_false) -> PlainQuantity | cs.MX:
    if any_is_qty(if_true, if_false):
        return if_true + if_false
    else:
        return cs.if_else(cond, if_true, if_false)


def safe_sign(x):
    if isinstance(x, PlainQuantity):
        return x
    else:
        return cs.sign(x)


def safe_sum(x):
    return (x**0).T @ x


def safe_cumsum(x):
    if isinstance(x, PlainQuantity):
        # For unit checks
        return x
    else:
        return cs.cumsum(x)


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


def thermo_deriv(eos, arg0, arg1, wrt: Literal[0, 1]):
    """
    wrt
    ---
        0 -> Derivative wrt to first arg
        1 -> Derivative wrt to second arg
    """
    # This is very cryptic and mysterious
    eos_value = (eos(arg0, arg1),)
    jacobian = eos.jacobian()(arg0, arg1, *eos_value)
    return [cs.diag(jacobian[wrt + 2 * i]) for i in range(eos.n_out())]


def thermo_fwd_fd(eos: CasadiEos, arg0, arg1, wrt: Literal[0, 1]):
    base_value = (eos(arg0, arg1),)

    delta_x = 1e-5
    if wrt == 0:
        delta_eos_value = (eos(arg0 + delta_x, arg1),)
    else:
        delta_eos_value = (eos(arg0, arg1 + delta_x),)

    return [
        (delta - base) / delta_x for delta, base in zip(delta_eos_value, base_value)
    ]


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


def span_fin_diff(f, x, edge_order: Literal['first', 'second'] = 'second'):
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


def residual_debugger(
    equation: EquationBase,
    nodes: list[int],
    data: dict[VarSpec, NDArray],
) -> Mapping[str, NDArray | EquationBase]:
    """
    Transpose the local namespace and variables of a
    residual equation to the currently active shell.
    I think this is probably unsafe with untrusted code.
    """
    module = sys.modules[equation.__module__]

    out = {'self': equation, **vars(module)}
    for name, spec in zip(equation.arg_names, equation.arg_specs):
        out[name] = data[spec._at_node(nodes.index(spec.node))]

    return out
