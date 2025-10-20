from typing import Literal

import CoolProp as cp

NodeStatesNames = Literal['stc', 'tot', 'rlt', 'kin', 'oth']

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
