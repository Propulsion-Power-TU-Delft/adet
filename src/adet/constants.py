from fontTools.voltLib.ast import Enum
from typing import Literal, Union
from numpy.typing import NDArray

import CoolProp as cp

NodeStatesNames = Literal['stc', 'tot', 'rlt', 'kin', 'geo', 'oth']


class ThermoNamesCoolProp(Enum):
    Pressure = 'p'
    Temperature = 'T'
    Hmass = 'hmass'
    Umass = 'umass'
    Smass = 'smass'
    Dmass = 'rhomass'
    Cpmass = 'cpmass'
    Cvmass = 'cvmass'
    Pcrit = 'p_critical'
    Quality = 'Q'


COOLPROP_NAMES_MAP = {
    'p': 'P',
    'T': 'T',
    'Q': 'Q',
    'hmass': 'Hmass',
    'umass': 'Umass',
    'smass': 'Smass',
    'rhomass': 'Dmass',
    'cpmass': 'Cpmass',
    'cvmass': 'Cvmass',
    'p_critical': 'P_critical',
}

INVERSE_CP_NAMES_MAP = {v: k for k, v in COOLPROP_NAMES_MAP.items()}

_PAIR_SUFFIX = '_INPUTS'
COOLPROP_PAIRS: dict[int, str] = {
    getattr(cp, key): key[: -len(_PAIR_SUFFIX)]
    for key in dir(cp)
    if key.endswith(_PAIR_SUFFIX)
}

AdetArray = Union[
    NDArray,
    list[float],
    list[int],
    tuple[float, ...],
    tuple[int, ...],
    float,
    int,
]
