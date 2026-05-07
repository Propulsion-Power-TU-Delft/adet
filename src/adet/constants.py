from enum import Enum
from typing import Literal, Union
from numpy.typing import NDArray

import CoolProp as cp

NodeStatesNames = Literal['stc', 'tot', 'rlt', 'kin', 'geo', 'oth']


class CoolProperties(Enum):
    Press = 'p'
    Temp = 'T'
    Hmass = 'hmass'
    Umass = 'umass'
    Smass = 'smass'
    Dmass = 'rhomass'
    Cpmass = 'cpmass'
    Cvmass = 'cvmass'
    Pcrit = 'p_critical'
    Quality = 'Q'
    SpeedSound = 'speed_sound'
    Viscosity = 'viscosity'


COOLPROP_NAMES_MAP = {
    CoolProperties.Press.value: 'P',
    CoolProperties.Temp.value: 'T',
    CoolProperties.Quality.value: 'Q',
    CoolProperties.Hmass.value: 'Hmass',
    CoolProperties.Umass.value: 'Umass',
    CoolProperties.Smass.value: 'Smass',
    CoolProperties.Dmass.value: 'Dmass',
    CoolProperties.Cpmass.value: 'Cpmass',
    CoolProperties.Cvmass.value: 'Cvmass',
    CoolProperties.Pcrit.value: 'P_critical',
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
