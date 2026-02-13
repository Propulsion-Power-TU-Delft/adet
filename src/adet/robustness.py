import logging
import multiprocessing as mp
from typing import Iterable
import scipy.stats.qmc as qmc
from copy import deepcopy
import numpy as np
from rich.progress import track
import matplotlib.pyplot as plt
from adet.tools.loggers import setup_logger
from adet.solution import solve_root_problem

logger = logging.getLogger(__name__)
setup_logger(logger, logging.ERROR, logging.ERROR)

# ============================================================================
# CONFIGURATION: Select which example to test
# ============================================================================
# Options: 'air_supply', 'nasa_hecc', 'nasa_hecc_multi'
# WARN: Careful you need to take in and out the multi loop from main
TEST_CASE = 'nasa_hecc'
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
elif TEST_CASE == 'nasa_hecc_multi':
    from adet.examples.nasa_hecc import (
        solution_hecc_multi as SOLUTION,
        ntw_hecc as NETWORK,
        kn_hecc_multi as KNOWNS,
        bnd_hecc_multi as BOUNDS,
    )
else:
    raise ValueError(f'Unknown test case: {TEST_CASE}')

# Module-level globals - each worker process will create its own copy
# when importing this module (no pickling needed)
ROOTFINDER = NETWORK.system.make_rootfinder('ipopt')


def process_sample(sample, solution, perturbation, bounded: bool):
    # Use global rootfinder and deepcopy it for this sample
    rootfinder_cp = deepcopy(ROOTFINDER)
    sample_trans = np.atleast_2d(sample).T
    delta_x0 = qmc.scale(sample_trans, -perturbation, +perturbation)
    x0_perturbed = solution + delta_x0

    try:
        bounds = BOUNDS if bounded else None
        solution = solve_root_problem(
            rootfinder_cp, x0_perturbed, KNOWNS, bounds, suppress_output=True
        )
        # Check that the solution actually matches
        if not np.isclose(solution, SOLUTION).all():
            print('System converged to a spurious solution')
            return 0
        else:
            return 1

    except Exception as e:
        print(f'reduced system did not converge, error {e}')
        return 0


def test_robustness(
    solution,
    num_samples,
    perturbations: Iterable[float],
    bounded: bool,
):
    out = {pert: [] for pert in perturbations}

    with mp.Pool(NUM_PROCS) as pool:
        for pert in track(perturbations, 'Overall progress'):
            sampler = qmc.LatinHypercube(len(solution))
            samples = sampler.random(num_samples)

            multires = [
                pool.apply_async(
                    process_sample,
                    (s, solution, pert, bounded),
                )
                for s in samples
            ]
            for res in track(
                multires,
                f'|> Progress over samples for perturbation {pert}',
            ):
                try:
                    out[pert].append(res.get(timeout=55))
                except mp.TimeoutError:
                    out[pert].append(0)

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
    PERTURBATIONS = np.arange(0.0001, 0.7, 0.1)  # Start with 10% offset
    # PERTURBATIONS = np.linspace(0, 4, 11) + 0.1
    BOUNDED = True

    results = test_robustness(SOLUTION, SAMPLES, PERTURBATIONS, BOUNDED)
    success_rate = np.array([sum(res) / SAMPLES for res in results.values()])

    # Save
    bnd_suffix = 'bounded' if BOUNDED else 'unbounded'
    FILENAME = f'./figures/{TEST_CASE}_samples_{SAMPLES}' + '_idealgas_' + bnd_suffix
    out_array = np.stack([PERTURBATIONS, success_rate])
    np.save(FILENAME, out_array)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(
        PERTURBATIONS,
        success_rate * 100,
        '-o',
        # color='seagreen',
        color='orange',
    )
    plt.xticks(
        ticks=PERTURBATIONS.tolist(), labels=[f'{100 * o:.0f}%' for o in results.keys()]
    )
    plt.xlabel(f'Perturbation from converged solution, {SAMPLES} samples')
    plt.ylabel('Success rate')
    plt.ylim(0.0, 110.0)
    plt.grid(alpha=0.3, axis='y')
    plt.plot(PERTURBATIONS, 100 * np.ones(len(PERTURBATIONS)))
    plt.title(TEST_CASE)

    plt.savefig(FILENAME + '.svg')
