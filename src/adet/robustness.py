import logging
import multiprocessing as mp
import scipy.stats.qmc as qmc
import numpy as np
from rich.progress import track
import matplotlib.pyplot as plt
from adet.tools.loggers import setup_logger
from copy import deepcopy
from adet.assembly import solve_root_problem

# --------------------
from adet.examples.air_supply_compressor_design import (
    kn_des,
    bnd_des_is,
    solution_des_is,
    rootfinder_des_is,
)

ROOTFINDER = rootfinder_des_is
# --------------------
# --------------------
from adet.examples.nasa_hecc import (
    kn_hecc,
    bnd_hecc_is,
    solution_hecc_is,
    rootfinder_hecc_is,
)

ROOTFINDER = rootfinder_hecc_is
# --------------------


logger = logging.getLogger(__name__)
setup_logger(logger, logging.ERROR, logging.ERROR)


def process_sample(sample, solution, knowns, bounds, perturbation):
    rootfinder_copy = deepcopy(ROOTFINDER)
    sample_trans = np.atleast_2d(sample).T
    x0_perturbed = solution + solution * perturbation * (-1 + 2 * sample_trans)

    try:
        solve_root_problem(
            rootfinder_copy, x0_perturbed.tolist(), knowns, bounds, suppress_output=True
        )
        return 1
    except Exception:
        return 0


def test_robustness(solution, knowns, bounds, samples_multiplier, max_perturbation):
    num_samples = int(
        samples_multiplier
        * max(
            ROOTFINDER.sparsity_in(0).shape,
        )
    )

    out = {j: [] for j in range(1, max_perturbation)}
    with mp.Pool(NUM_PROCS) as pool:
        for pert in track(range(1, max_perturbation), 'Overall progress'):
            sampler = qmc.LatinHypercube(len(solution))
            samples = sampler.random(num_samples)

            multires = [
                pool.apply_async(
                    process_sample,
                    (s, solution, knowns, bounds, pert),
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

    for pert, results in out.items():
        plt.bar(pert, 100 * sum(results) / num_samples, color='b')

    print(f'Success rate for perturbation rate {pert} {sum(results) / num_samples}')
    plt.xticks(ticks=list(out.keys()), labels=[f'{100 * o}%' for o in out.keys()])
    plt.xlabel('Perturbation from converged solution')
    plt.ylabel('Convergence rate')
    plt.show()


if __name__ == '__main__':
    mp.freeze_support()
    # Latin Hypercube sampling for testing robustness
    SAMPLES_MULTIPLIER = 3
    NUM_PROCS = 10
    MAX_PERT = 8

    # test_robustness(solution_des_is, kn_des, bnd_des_is, SAMPLES_MULTIPLIER, MAX_PERT)

    test_robustness(
        solution_hecc_is, kn_hecc, bnd_hecc_is, SAMPLES_MULTIPLIER, MAX_PERT
    )
