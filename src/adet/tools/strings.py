"""
Basic tools for string manipulation
"""

import re
from typing import get_args, cast
from adet.constants import NodeStatesNames

STATE_NAMES = get_args(NodeStatesNames)


def split_by_uppercase(arg: str):
    return re.findall(r'[A-Z][a-z]*', arg)


def rm_digits(argument: str) -> str:
    """
    Remove digits from the argument
    """
    # \D = NOT digits
    variable_string = re.findall(r'^\D+', argument)
    if len(variable_string) > 1:
        raise ValueError(
            f'Badly formatted argument: {variable_string}. Please provide'
            f'the indices at the end e.g. `kine_V0`.'
        )

    return variable_string[0]


def get_index(argument: str) -> int:
    """
    Get an index from a string, where the index is supposed to
    be at the end of the string formatted as <string><index>
    """
    # \d = digits
    string_index = re.findall(r'\d+$', argument)

    if len(string_index) > 1:
        raise ValueError(
            f'Badly formatted argument: {string_index}. Please provide'
            f'the indices at the end e.g. `kine_V0`.'
        )

    return int(string_index[0])


def verify_string_pattern(argument: str, reference_pattern: str) -> bool:
    """Verify whether the string argument satisfies exactly a reference pattern"""
    regex_match = re.findall(reference_pattern, argument)
    if not regex_match:
        return False
    else:
        if len(regex_match) == 1:
            return True
        else:
            return False


def get_arg_type(argument: str, prefix_length: int = 4) -> str:
    """
    Isolate the var type, removing digits and prefixes
    """
    var_type = rm_digits(argument)[prefix_length:]
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
        raise ValueError(f'Unknown variable type {state_id}')
