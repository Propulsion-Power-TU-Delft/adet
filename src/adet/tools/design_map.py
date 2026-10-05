"""Generic 2D design-map sweep driver.

Sweeps a square grid of two boundary-condition ``VarSpec``s over an
already-built ``ComponentNetwork``, warm-starting each point from its
nearest already-solved neighbor.
"""

import logging
from typing import Callable

import numpy as np

from adet.assemblers import CasadiSystem
from adet.components import ComponentNetwork
from adet.solution import solve_root_problem
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)


def sweep_design_map(
    ntw: ComponentNetwork[CasadiSystem],
    x_spec: VarSpec,
    y_spec: VarSpec,
    x_span: np.ndarray,
    y_span: np.ndarray,
    first_sol,
    starter_keys=None,
    starter_solutions=None,
    custom_bounds: dict | None = None,
    validity_check: Callable[[dict], bool] | None = None,
):
    """
    Sweep ``x_spec``/``y_spec`` over the square ``x_span`` x ``y_span``
    grid, solving the already-built network ``ntw`` at each point starting
    from its nearest already-solved neighbor (kinsol first, falling back
    to bounded/unbounded ipopt, then the starting guess itself).

    Parameters
    ----------
    first_sol : the solution the first grid point warm-starts from.
    starter_keys, starter_solutions : optional pre-computed (key,
        solution) pairs to warm-start every point from instead of the
        running solution cache.
    custom_bounds : passed to ``ntw.system.get_bounds``.
    validity_check : optional ``sol_dict -> bool`` (True = keep the point),
        applied in addition to the standard NaN check.
    """
    if len(x_span) != len(y_span):
        raise ValueError(
            'x_span and y_span must be the same length (a square sweep grid)'
        )
    n_points = len(x_span)

    rtfn_kin = ntw.system.make_rootfinder('kinsol')
    rtfn_ip = ntw.system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': True,
            'ipopt.max_wall_time': 1.0,
        },
    )

    keys = np.zeros((n_points**2, 2))

    solutions = np.zeros((n_points**2, len(first_sol)))
    solutions[0] = first_sol.flatten()

    # Store solution dicts for each point
    solution_dicts = []

    kn = ntw.system.get_boundary_conds()
    bnd = ntw.system.get_bounds(custom_bounds=custom_bounds or {})

    curr_index = 0
    boun_cond_keys = list(ntw.system.data.boun_cond.keys())
    x_idx = boun_cond_keys.index(x_spec)
    y_idx = boun_cond_keys.index(y_spec)
    for x_val in x_span:
        for y_val in y_span:
            curr_key = np.array([x_val, y_val])
            if starter_keys is not None and starter_solutions is not None:
                distances = np.linalg.norm(curr_key - starter_keys, axis=1, ord=2)
                idx = np.argmin(distances)
                x0 = ntw.system.get_guess(starter_solutions[idx])
            else:
                distances = np.linalg.norm(curr_key - keys, axis=1, ord=np.inf)
                idx = np.argmin(distances)
                x0 = solutions[idx]
                while np.isnan(x0).any():
                    idx -= 1
                    logger.warning('Solution cache miss, going to best next one')
                    x0 = solutions[idx]

            kn[x_idx] = np.array([x_val * ntw.system.constraints_scaling[x_idx]])
            kn[y_idx] = np.array([y_val * ntw.system.constraints_scaling[y_idx]])
            solution = x0
            try:
                solution = solve_root_problem(rtfn_kin, x0, kn, suppress_output=True)
            except RuntimeError:
                logger.info('KINSOL failed, trying IPOPT...')
                try:
                    solution = solve_root_problem(
                        rtfn_ip, x0, kn, bnd, suppress_output=True
                    )
                except RuntimeError:
                    try:
                        solution = solve_root_problem(
                            rtfn_ip, x0, kn, suppress_output=True
                        )
                    except RuntimeError:
                        logger.info('IPOPT failure, default to closest solution')
                        solution = x0

            keys[curr_index, :] = curr_key
            solutions[curr_index, :] = np.array(solution).flatten()

            if not np.isnan(solution).any():
                sol_dict = ntw.system.sol_to_dict(solution)
                if validity_check is not None and not validity_check(sol_dict):
                    sol_dict = None
                    logger.warning(f'Failed point at x={x_val:.2f}, y={y_val:.2f}')
            else:
                sol_dict = None
                logger.warning(f'Failed point at x={x_val:.2f}, y={y_val:.2f}')

            solution_dicts.append(sol_dict)

            curr_index += 1

    return keys, solutions, solution_dicts
