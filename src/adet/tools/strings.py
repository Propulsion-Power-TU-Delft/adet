"""
Basic tools for string manipulation
"""

import re
from typing import cast, get_args

from adet.constants import NodeStatesNames

STATE_NAMES = get_args(NodeStatesNames)


def split_by_uppercase(arg: str):
    return re.findall(r'[A-Z][a-z]*', arg)


def rm_index(argument: str) -> str:
    """
    Remove digits from the argument
    """
    # \D = NOT digits
    variable_string = re.split(r'\d+$', argument)
    return variable_string[0]


def change_idx(argument: str, new_idx: int):
    return re.sub(r'\d+$', str(new_idx), argument)


def get_index(argument: str) -> int:
    """
    Get an index from a string, where the index is supposed to
    be at the end of the string formatted as <string><index>
    """
    string_index = re.findall(r'\d+$', argument)

    if not string_index:
        raise AttributeError(f'No digits in `{argument}`')

    return int(string_index[0])


def validate_arg_format(argument: str, include_digits: bool):
    states_id_re = '|'.join(get_args(NodeStatesNames))
    digits = r'\d+$' if include_digits else ''
    PATTERN = rf'^({states_id_re})_[a-zA-Z0-9_]*' + digits
    return re.match(PATTERN, argument)


def get_arg_type(argument: str, prefix_length: int = 4) -> str:
    """
    Isolate the var type, removing digits and prefixes
    """
    var_type = rm_index(argument)[prefix_length:]
    return var_type


def get_arg_state(argument: str, prefix_length: int = 3) -> NodeStatesNames:
    """
    Isolate the var state prefix (a.k.a.) the first
    """
    state_id = argument[:prefix_length]

    if state_id in STATE_NAMES:
        state_id = cast(NodeStatesNames, state_id)
        return state_id
    else:
        raise ValueError(
            f'Unknown state for `{argument}`, valid states are:\n{STATE_NAMES}'
        )


def get_arg_specs(argument: str) -> tuple[NodeStatesNames, str, int]:
    arg_idx = get_index(argument)
    arg_type = get_arg_type(argument)
    arg_state = get_arg_state(argument)
    return arg_state, arg_type, arg_idx
