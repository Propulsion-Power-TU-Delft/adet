"""
Miscellaneous of string manipulation and helper functions
to interact with the CoolProp library
"""

import logging
import re
from math import ceil
from typing import Sequence
import CoolProp.CoolProp as cp
import numpy as np

from numpy.typing import NDArray
import matplotlib.pyplot as plt

from adet.tools.interpolation import TransfiniteInterpolator
from adet.constants import COOLPROP_NAMES_MAP, COOLPROP_PAIRS, INVERSE_CP_NAMES_MAP

cp_INPUTS_SUFFIX = '_INPUTS'

logger = logging.getLogger(__name__)


def get_input_names(input_pair: int) -> list[str]:
    """
    Example
    -------
    >>> get_input_names(cp.PT_INPUTS)
    ['P', 'T']
    """
    pair_name = COOLPROP_PAIRS[input_pair]
    # Split based on capital letters
    return re.findall(r'[A-Z][a-z]*', pair_name)


def pair_based_sorting(*update_variables: str) -> tuple[str, ...]:
    """
    Example
    -------
    Since `rhomass` corresponds to `Dmass`, it comes before `Smass`:
    >>> pair_sorting('smass', 'rhomass')
    ('rhomass', 'smass')
    """
    return tuple(
        sorted(
            update_variables,
            key=lambda x: COOLPROP_NAMES_MAP[x],
        )
    )


def pair_name_from_tuple(update_variables: tuple[str, ...]):
    """
    Convert a tuple, e.g. ('p', 'T') to the correspondent
    CoolProp pair, e.g. PT_INPUTS
    """
    sorted_pair = pair_based_sorting(*update_variables)
    mapped_pair = (COOLPROP_NAMES_MAP[var] for var in sorted_pair)
    return ''.join(mapped_pair)


def pair_id_from_name(input_pair_name: str) -> int:
    """
    Get the pair id from its name, e.g. HmassSmass -> 26
    """
    input_pairs = {
        attr[: -len(cp_INPUTS_SUFFIX)]: getattr(cp, attr)
        for attr in dir(cp)
        if attr.endswith(cp_INPUTS_SUFFIX)
    }

    return input_pairs[input_pair_name]


def pair_id_from_tuple(update_variables: tuple[str, ...]):
    pair_name = pair_name_from_tuple(update_variables)
    return pair_id_from_name(pair_name)


def pair_tuple_from_id(input_pair: int) -> list[str]:
    input_names = get_input_names(input_pair)
    return [INVERSE_CP_NAMES_MAP[inp] for inp in input_names]


def make_lookup_table(
    out_pties: str | Sequence[str],
    input_pair: str,
    grid: NDArray,
):
    if isinstance(out_pties, str):
        out_pties = [out_pties]

    # Convert string name -> CoolProp constant
    pair_id = pair_id_from_name(input_pair)

    nu, nv, _ = grid.shape
    luts = {pty: np.full((nu, nv), np.nan) for pty in out_pties}

    for i in range(nu):
        for j in range(nv):
            v0, v1 = grid[i, j]
            try:
                AS.update(pair_id, v0, v1)
                for pty in out_pties:
                    luts[pty][i, j] = getattr(AS, pty)()
            except Exception:
                continue

    return luts


class DebugAbstractState(cp.AbstractState):
    """
    Light wrapper for counting the number of updates
    and printing on update
    """

    def __init__(self, *args, **kwargs) -> None:
        self.num_updates = 0
        self.debug_print = False
        super().__init__()

    def update(self, *args, **kwargs):
        self.num_updates += 1
        debug_str = f"""
========================
|> UPDATE DEBUG
   ------------
|> Updating with {COOLPROP_PAIRS[args[0]]}
|> First arg: {args[1]}
|> Second arg: {args[2]}
========================
        """

        if self.debug_print:
            print(debug_str)
        return super().update(*args, **kwargs)


def plot_contours(grid, luts, properties, levels=20, cmap='viridis'):
    """
    Plot contour maps of lookup table properties.

    Parameters
    ----------
    grid : (nu,nv,2) array
        Grid of state points.
    luts : dict
        Dictionary {property: ndarray(nu,nv)} from make_lookup_table().
    properties : str or list[str]
        Properties to plot.
    levels : int or sequence
        Number of contour levels or explicit levels.
    cmap : str
        Colormap for contours.
    """
    if isinstance(properties, str):
        properties = [properties]

    X = grid[:, :, 0]
    Y = grid[:, :, 1]

    nplots = len(properties)
    _, axes = plt.subplots(
        ceil(nplots / 3),
        3,
        figsize=(3 * nplots, 8),
        squeeze=False,
    )

    for ax, pty in zip(axes.flat, properties):
        Z = luts[pty]

        cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label(pty)

        ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.5, alpha=0.6)

        ax.set_xlabel('Input 1')
        ax.set_ylabel('Input 2')
        ax.set_title(f'Contour of {pty}')

    plt.tight_layout()
    plt.show()


def calc_saturation_curves(
    eos: cp.AbstractState, n_points: int = 500
) -> tuple[dict[str, NDArray], ...]:
    """
    Calculate all the saturation curves in one shot using
    the state class to save computational time.
    * Vary Q discretely between 0 and 1 to obtain the properties
    in the liquid and vapor phase
    * Sweep between T_triple and T_critical
    """
    dictL, dictV = {}, {}  # Liquid and Vapour
    for Q, dic in zip([0, 1], [dictL, dictV]):
        rhomass, smass, hmass, T, p, umass = [], [], [], [], [], []
        for _T in np.logspace(
            np.log10(eos.keyed_output(cp.iT_triple)),
            np.log10(eos.keyed_output(cp.iT_critical)),
            n_points,
        ):
            try:
                eos.update(cp.QT_INPUTS, Q, _T)
                if eos.p() < 0:
                    raise ValueError('P is negative:' + str(eos.p()))

                T.append(eos.T())
                p.append(eos.p())
                rhomass.append(eos.rhomass())
                hmass.append(eos.hmass())
                smass.append(eos.smass())
                umass.append(eos.umass())
            except ValueError:
                pass

        dic.update(
            dict(
                T=np.array(T),
                p=np.array(p),
                rhomass=np.array(rhomass),
                hmass=np.array(hmass),
                smass=np.array(smass),
                umass=np.array(umass),
            )
        )

    return dictL, dictV


def calc_Tmax_curves(eos: cp.AbstractState, n_points: int = 500) -> dict[str, NDArray]:
    """
    Fix T=Tmax, loop over the pressure, from iPmin to iPmax
    to obtain the thermodynamic quantities along the Tmax curve
    """
    rhomass, smass, hmass, T, p, umass = [], [], [], [], [], []

    for _p in np.logspace(
        np.log10(eos.keyed_output(cp.iP_min) * 1.01),
        np.log10(eos.keyed_output(cp.iP_max)),
        n_points,
    ):
        try:
            eos.update(cp.PT_INPUTS, _p, eos.keyed_output(cp.iT_max))
        except ValueError:
            continue

        try:
            T.append(eos.T())
            p.append(eos.p())
            rhomass.append(eos.rhomass())
            hmass.append(eos.hmass())
            smass.append(eos.smass())
            umass.append(eos.umass())
        except ValueError:
            pass

    Tmax = dict(
        T=np.array(T),
        p=np.array(p),
        rhomass=np.array(rhomass),
        hmass=np.array(hmass),
        smass=np.array(smass),
        umass=np.array(umass),
    )

    return Tmax


if __name__ == '__main__':
    AS = cp.AbstractState('HEOS', 'MM')
    N = 50
    exclude = N // 10  # Cut head and tail
    N_corrected = N - 2 * exclude

    out_pties = [
        'p',
        'T',
        'hmass',
        'smass',
        'rhomass',
        'umass',
    ]

    INPUT_PAIR = pair_based_sorting('rhomass', 'smass')
    pair_name = pair_name_from_tuple(INPUT_PAIR)

    _, sat_curves = calc_saturation_curves(AS, N_corrected)
    max_curves = calc_Tmax_curves(AS, N_corrected)

    sat_curve = np.vstack(
        [
            sat_curves[INPUT_PAIR[0]][exclude:-exclude],
            sat_curves[INPUT_PAIR[1]][exclude:-exclude],
        ]
    )

    max_curve = np.vstack(
        [
            max_curves[INPUT_PAIR[0]][exclude:-exclude],
            max_curves[INPUT_PAIR[1]][exclude:-exclude],
        ]
    )

    interpolator = TransfiniteInterpolator(
        sat_curve, max_curve, N_corrected, N_corrected
    )
    grid = interpolator.generate_grid()
    interpolator.plot()

    luts = make_lookup_table(out_pties, pair_name, grid)

    plot_contours(grid, luts, out_pties, 50, 'viridis')
