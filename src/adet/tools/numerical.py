"""
Numerical methods implementation
"""

import logging
from collections import namedtuple
from typing import Callable

import jax.numpy as jnp

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
