import logging
import multiprocessing as mp
import scipy.stats.qmc as qmc
import numpy as np

from copy import deepcopy
from adet.assembly import solve_root_problem
from adet.examples.air_supply_compressor_design import (
    kn,
    bnd_is,
    solution_is,
    rootfinder_is,
)
import matplotlib.pyplot as plt

from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger, logging.ERROR, logging.ERROR)


def process_sample(sample, perturbation):
    rootfinder = deepcopy(rootfinder_is)
    sample_trans = np.atleast_2d(sample).T
    x0_perturbed = solution_is + solution_is * perturbation * (-1 + 2 * sample_trans)

    try:
        solve_root_problem(
            rootfinder, x0_perturbed.tolist(), kn, bnd_is, suppress_output=True
        )
        return 1
    except Exception:
        return 0


if __name__ == '__main__':
    from rich.progress import track

    mp.freeze_support()
    # Latin Hypercube sampling for testing robustness
    NUM_SAMPLES = 100
    NUM_PROCS = 10
    MAX_PERT = 10

    out = {j: [] for j in range(0, MAX_PERT)}
    with mp.Pool(NUM_PROCS) as pool:
        for pert in track(range(0, MAX_PERT), 'Overall progress', MAX_PERT):
            sampler = qmc.LatinHypercube(len(solution_is))
            samples = sampler.random(NUM_SAMPLES)

            multires = [pool.apply_async(process_sample, (s, pert)) for s in samples]
            out[pert] = [
                res.get(timeout=60)
                for res in track(
                    multires,
                    f'|> Progress over samples for perturbation {pert}',
                )
            ]

    for pert, results in out.items():
        plt.bar(pert, sum(results) / NUM_SAMPLES)

    plt.show()
