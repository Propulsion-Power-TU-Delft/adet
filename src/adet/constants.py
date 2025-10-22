from typing import Literal, Union
from numpy.typing import NDArray

import CoolProp as cp

NodeStatesNames = Literal['stc', 'tot', 'rlt', 'kin', 'geo', 'oth']

COOLPROP_NAMES_MAP = {
    'p': 'P',
    'T': 'T',
    'Q': 'Q',
    'hmass': 'Hmass',
    'umass': 'Umass',
    'smass': 'Smass',
    'rhomass': 'Dmass',
}

_SUFFIX = '_INPUTS'
COOLPROP_PAIRS = {
    getattr(cp, key): key[: -len(_SUFFIX)] for key in dir(cp) if key.endswith(_SUFFIX)
}

ArrayLike = Union[
    NDArray,
    list[float],
    list[int],
    tuple[float, ...],
    tuple[int, ...],
    float,
    int,
]
