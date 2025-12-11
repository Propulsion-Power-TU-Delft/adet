"""
Numerical methods implementation
"""

import logging
from typing import Callable
from collections import namedtuple
import casadi as cs

import jax.numpy as jnp
import jax


logger = logging.getLogger(__name__)

NewtonSolution = namedtuple('NewtonSolution', ('value', 'fun', 'jac', 'nit', 'success'))


def newton_method(
    res_func: Callable,
    x0,
    jac_func: Callable,
    xtol: float = 1e-4,
    atol: float = 1e-2,
    max_iter: int = 100,
    rel_factor: float = 1.0,
    callback: Callable | None = None,
) -> NewtonSolution:
    """
    Pure Newton method - basic implementation with automatic
    relaxation and least square linear system solution for near-singular jacobians

    Parameters
    ----------
    res_func: Callable,
        Function that returns the residuals as a 0-D array
    x0,
        Initial guess
    jac_func: Callable,
        Function that returns the Jacobian matrix
    xtol: float = 1e-4,
        Tolerance for relative solution variation between successive iterations
    atol: float = 1e-2,
        Absolute tolerance for residual function values
    max_iter: int = 100,
        Maximum iterations allowed
    rel_factor: float = 1.0,
        Relaxation factor in the newton step, it is automatically adjusted if the
        solution moves outside of the domain of the residual functions
    callback: Callable | None = None,
        Function called at each iteration on the solution object, e.g. can be used
        for convergence analysis and storage
    """

    # Initialize
    x = x0
    success = False

    for nit in range(1, max_iter):
        res = res_func(x)
        jac = jac_func(x)

        # Automatic relaxation step if the residual is NaN
        if jnp.isnan(res).any():
            if rel_factor >= 0.3:
                rel_factor -= 0.2
                logger.debug(
                    f'NaN present in N-R iteration residuals, '
                    f'relaxing...  (factor = {rel_factor:.1f})'
                )
                x = x0
                continue
            else:
                logger.debug('NaN still encountered after all relaxation steps.')
                success = False
                break

        dx = jnp.linalg.solve(jac, -res)

        if jnp.isnan(dx).any():
            logger.debug('NaN in N-R solution variation, trying with least squares...')
            dx, _, _, _ = jnp.linalg.lstsq(jac, -res)

        # NOTE: We rely on this to overwrite x,
        # otherwise x and x0 would have the same pointer, careful!
        x += rel_factor * dx

        rel_success = (abs(dx) < xtol * (1 + abs(x))).all()
        abs_success = max(abs(res)) < atol

        success = rel_success and abs_success

        solution = NewtonSolution(x, res, jac, nit, success)

        # User-defined actions on the solution at the n-th iteration
        if callback:
            callback(solution)

        if success:
            break

    if not success:
        logger.info('The Newton-Raphson solver did not converge')
    else:
        logger.info('The Newton-Raphson solver converged successfully')

    return solution


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


def derivative(x0, func, eps, argnum):
    """
    Compute the derivative of `func` w.r.t. argument
    number `argnum` at `x0`, using forward finite
    differences with step `eps`
    """
    x1 = x0.at[argnum].set(x0[argnum] + eps)
    return (func(x1) - func(x0)) / eps


def jacfdiff(func, eps: float = 1e-5):
    """
    Create a callable function for a jacobian computation,
    works by vmapping derivative over the argument number
    """

    def jacobian_partial(x0):
        num_args = x0.shape[0]
        argnums = jnp.arange(num_args)

        jacobian = jax.vmap(
            derivative,
            in_axes=(None, None, None, 0),
            out_axes=1,
        )
        return jacobian(x0, func, eps, argnums)

    return jax.vmap(jacobian_partial, 1, 0)


if __name__ == '__main__':
    # Example of a custom jacobian vector product
    def func_raw(X):
        x, y = X
        return jnp.array(
            [
                jnp.sin(x) * y,
                # jnp.cos(y) + x,
            ]
        )

    @jax.custom_jvp
    def func_wrap(X):
        return func_raw(X)

    @func_wrap.defjvp
    def func_vjp(primals, tangents):
        (X,) = primals
        (X_dot,) = tangents

        jacfd = jacfdiff(func_wrap.__wrapped__, 1e-5)

        primal_out = func_wrap.__wrapped__(X)
        batch_matmul = jax.vmap(jnp.matmul, (0, 1), 1)

        tangent_out = batch_matmul(jacfd(X), X_dot)

        return primal_out, tangent_out

    x0 = jnp.array(
        [
            [0.2],  # , 0.3, 0.4],
            [0.1],  # , 0.2, 0.3],
        ]
    )

    jac_raw = jax.jit(jax.jacrev(func_raw))
    jac_custom = jax.jit(jax.jacrev(func_wrap))

    jac_ad_val = jac_raw(x0)
    jac_custom_val = jac_custom(x0)

    print(f'Original jacrev\n{jac_ad_val}\n')
    print(f'FD jacrev\n{jac_custom_val}\n')
