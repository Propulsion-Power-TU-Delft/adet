from functools import partial
from adet.assemblers import CasadiSystem, IPOPT_DEFAULTS
import logging
from typing import Any, Callable

import casadi as cs
import numpy as np
from numpy.typing import NDArray

from adet.tools.context import dummy_context, output_suppression

logger = logging.getLogger(__name__)


# NOTE: I am using this instead of qmc because
# scipy somehow breaks pdb on windows
def _latin_hypercube(n_dims, n_samples):
    result = np.zeros((n_samples, n_dims))
    for i in range(n_dims):
        perm = np.random.permutation(n_samples)
        result[:, i] = (perm + np.random.uniform(size=n_samples)) / n_samples
    return result * 2 - 1  # scale [0, 1] -> [-1, 1]


def generate_perturbated_samples(guess, num_samples, delta_pert):
    samples = _latin_hypercube(len(guess), num_samples) * delta_pert

    if isinstance(guess, list):
        return np.concatenate(guess).flatten() + samples
    else:
        return guess + samples


def multi_solver(rootfinder, guess, knowns, bounds, delta_pert, num_samples):
    guesses = generate_perturbated_samples(guess, num_samples, delta_pert)

    for x0 in guesses:
        try:
            solution = solve_root_problem(
                rootfinder,
                x0,
                knowns,
                bounds,
                suppress_output=True,
            )
        except RuntimeError:
            continue

        if solution:
            break

    return solution


def best_first_iter(guess, knowns, root_function, delta_pert, num_samples):
    guesses = generate_perturbated_samples(guess, num_samples, delta_pert)

    def norm_function(x):
        return np.linalg.norm(x, np.inf)

    best_guess = guess
    best_res_norm = norm_function(
        root_function(guess, knowns),
    )

    logger.info(f'Trying out {num_samples} latin hypercube samples for first guess...')
    for x0 in guesses:
        # Perturb the original guess
        # Compute first iteration residual
        try:
            initial_residual = root_function(x0, knowns)
        except RuntimeError:
            continue
        # Compute the residual norm
        residual_norm = norm_function(initial_residual)
        # If the norm is better the the current one, write that guess
        if residual_norm < best_res_norm:
            logger.info(
                f'Found better initial guess norm {residual_norm:.3f} '
                f'< {best_res_norm:.3f}'
            )
            best_guess = x0
            best_res_norm = residual_norm

    if (best_guess == guess).all():
        logger.info('No better solution found, using random perturbation')
        best_guess = x0 + delta_pert * np.random.ranf(len(x0))

    return best_guess


def solve_root_problem(
    rootfinder: Any,
    guess: list[NDArray] | NDArray,
    knowns: list[NDArray],
    arg_bounds: tuple[cs.DM, cs.DM] | None = None,
    suppress_output: bool = False,
    *,
    # Guess perturbation
    perturbate_guess: bool = False,
    delta_pert: float = 0.02,
    num_samples: int = 100,
):
    """Simple utility function for solving rootfinding problems"""
    if suppress_output:
        output_manipulator = output_suppression
    else:
        output_manipulator = dummy_context

    with output_manipulator():
        logger.info(f'Solving the system with {rootfinder.name()}...')

        if isinstance(guess, list):
            guess_cat = np.concatenate(guess)
        else:
            guess_cat = guess

        knowns_cat = np.concatenate(knowns)

        if perturbate_guess:
            if rootfinder.n_in() < 2:
                raise NotImplementedError('Perturbation only implemented for ipopt')

            root_fn = rootfinder.get_function('nlp_g')
            guess_cat = best_first_iter(
                guess_cat, knowns_cat, root_fn, delta_pert, num_samples
            )

        extra_args: dict[str, int | cs.DM] = {}
        if rootfinder.n_in() > 2:
            extra_args.update({'lbg': 0, 'ubg': 0})
            if arg_bounds:
                extra_args['lbx'], extra_args['ubx'] = arg_bounds

        sol = rootfinder(
            x0=guess_cat,
            p=knowns_cat,
            **extra_args,
        )

        if isinstance(sol, dict):
            sol = sol['x']

        return np.array(sol)


def solve_optimization_problem(
    system: CasadiSystem,
    obj_func: cs.MX | float,
    guess: list[NDArray] | NDArray,
    knowns: list[NDArray],
    arg_bounds: tuple[cs.DM, cs.DM] | None = None,
    opts={},
) -> tuple[
    dict[str, cs.DM],
    Callable[[Any, Any], dict[str, cs.DM]],
]:
    """
    Returns
    -------
    solution
        The dictionary of specs of the solution
    partial_opt
        Partialized optimizer that takes the guess and
        knowns vectors as inputs
    """

    args_sym = list(system.free_args_sym.values())
    cons_sym = list(system.const_sym.values())

    free_args_symbols = cs.vertcat(*args_sym)
    constraints_symbols = cs.vertcat(*cons_sym)

    res_func = system.make_residual_function()
    res_expr = res_func(free_args_symbols, constraints_symbols)

    opt_problem = {
        'x': free_args_symbols,
        'p': constraints_symbols,
        'f': obj_func,
        'g': res_expr,
    }
    optimizer = cs.nlpsol(
        'optimizer',
        'ipopt',
        opt_problem,
        {**IPOPT_DEFAULTS, **opts},
    )

    x0 = np.concatenate(guess)
    kn = np.concatenate(knowns)

    kwargs_opt = {}
    kwargs_opt.update(
        {
            # Force the root problem
            'lbg': 0,
            'ubg': 0,
        }
    )

    if arg_bounds:
        kwargs_opt.update(
            {
                'lbx': arg_bounds[0],
                'ubx': arg_bounds[1],
            }
        )

    solution = optimizer(x0=x0, p=kn, **kwargs_opt)

    def partial_opt(x0, kn):
        return optimizer(x0=x0, p=kn, **kwargs_opt)

    return solution, partial_opt
