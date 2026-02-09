import logging
import multiprocessing as mp
from typing import Iterable
import scipy.stats.qmc as qmc
from copy import deepcopy
import numpy as np
from rich.progress import track
import matplotlib.pyplot as plt
from adet.tools.loggers import setup_logger
from adet.assembly import solve_root_problem

logger = logging.getLogger(__name__)
setup_logger(logger, logging.ERROR, logging.ERROR)

# ============================================================================
# CONFIGURATION: Select which example to test
# ============================================================================
TEST_CASE = 'air_supply'  # Options: 'air_supply', 'nasa_hecc', etc.
# ============================================================================

# Import the appropriate example based on configuration
if TEST_CASE == 'air_supply':
    from adet.examples.air_supply_compressor_design import (
        solution_ass_is as SOLUTION,
        ntw_ass as NETWORK,
        kn_ass as KNOWNS,
        bnd_ass_is as BOUNDS,
    )
elif TEST_CASE == 'nasa_hecc':
    from adet.examples.nasa_hecc import (
        solution_hecc_is as SOLUTION,
        ntw_hecc as NETWORK,
        kn_hecc_is as KNOWNS,
        bnd_hecc_is as BOUNDS,
    )
else:
    raise ValueError(f'Unknown test case: {TEST_CASE}')

# Module-level globals - each worker process will create its own copy
# when importing this module (no pickling needed)
ROOTFINDER = NETWORK.system.make_rootfinder('ipopt')


def process_sample(
    sample,
    solution,
    perturbation,
):
    # Use global rootfinder and deepcopy it for this sample
    rootfinder_cp = deepcopy(ROOTFINDER)
    sample_trans = np.atleast_2d(sample).T
    delta_x0 = qmc.scale(sample_trans, -perturbation, +perturbation)
    x0_perturbed = solution + delta_x0

    try:
        solve_root_problem(
            rootfinder_cp, x0_perturbed.tolist(), KNOWNS, BOUNDS, suppress_output=True
        )
        return 1
    except Exception:
        return 0


def test_robustness(
    solution,
    num_samples,
    perturbations: Iterable[float],
):
    out = {pert: [] for pert in perturbations}

    with mp.Pool(NUM_PROCS) as pool:
        for pert in track(perturbations, 'Overall progress'):
            sampler = qmc.LatinHypercube(len(solution))
            samples = sampler.random(num_samples)

            multires = [
                pool.apply_async(
                    process_sample,
                    (s, solution, pert),
                )
                for s in samples
            ]
            out[pert] = [
                res.get()
                for res in track(
                    multires,
                    f'|> Progress over samples for perturbation {pert}',
                )
            ]

            print(
                f'Success rate for perturbation rate {pert} '
                f'{sum(out[pert]) / num_samples}'
            )

    return out


if __name__ == '__main__':
    mp.freeze_support()
    # Latin Hypercube sampling for testing robustness
    SAMPLES = 100  # For each perturbation level
    NUM_PROCS = 10
    PERTURBATIONS = np.linspace(0, 3, 11) + 0.05  # Start with 5% offset

    results = test_robustness(SOLUTION, SAMPLES, PERTURBATIONS)

    plt.plot(
        PERTURBATIONS,
        [100 * sum(res) / SAMPLES for res in results.values()],
        '-o',
    )
    plt.xticks(
        ticks=PERTURBATIONS.tolist(), labels=[f'{100 * o:.0f}%' for o in results.keys()]
    )
    plt.xlabel(f'Perturbation from converged solution, {SAMPLES} samples')
    plt.ylabel('Convergence rate')
    plt.ylim(0.0, 110.0)
    plt.grid(alpha=0.3, axis='y')
    plt.plot(PERTURBATIONS, 100 * np.ones(len(PERTURBATIONS)))
    plt.title(TEST_CASE)
    plt.show()
