from typing import Type
from pathlib import Path
import subprocess
import importlib.util

# Explicitly load the .py implementation, bypassing stub resolution
var_hinting_path = Path(__file__).parent.parent / 'equations' / 'var_hinting.py'
spec = importlib.util.spec_from_file_location('var_hinting_impl', var_hinting_path)

if spec is None or spec.loader is None:
    raise ModuleNotFoundError(f'Hinting module not found at {var_hinting_path}')

var_hinting_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(var_hinting_module)

from adet.equations.var_hinting import (
    ThermoVariables,
    ThermoHints,
    VarSpec,
    VariableHints,
    OtherVariables,
    OtherHints,
    NodeHints,
)


# Imports
lines = [
    'from enum import Enum',
    'from casadi import MX',
    'from pint import Quantity',
    'from typing import Annotated, Type',
]

# Simple classes
lines.extend(
    [
        f'class {VarSpec.__name__}: ...',
        f'class {VariableHints.__name__}: ...',
        f'class {OtherVariables.__name__}(Enum): ...',
        f'class {ThermoVariables.__name__}(Enum): ...',
    ],
)


def generate_hint_class(
    hint_class: Type[VariableHints],
    enum_class: Type[ThermoVariables | OtherVariables],
):
    class_lines = []
    class_lines.append(f'class {hint_class.__name__}({VariableHints.__name__}):')
    class_lines.append('    def __init__(self, prefix: str): ...')
    for var in enum_class:
        class_lines.append('    @property')
        class_lines.append(
            f'    def    {var.name}(self) '
            f'-> Type[Annotated[MX | Quantity, {VarSpec.__name__}]]: ...'
        )
    return class_lines


def generate_node_hints():
    class_lines = [
        f'class {NodeHints.__name__}:',
        '    def __init__(self, index: int): ...',
    ]

    for meth in dir(NodeHints):
        if meth.startswith('_'):
            continue

        class_lines.extend(
            [
                '    @property',
                f'    def {meth}(self) -> {ThermoHints.__name__}: ...',
            ]
        )

    return class_lines


lines.extend(generate_hint_class(ThermoHints, ThermoVariables))
lines.extend(generate_hint_class(OtherHints, OtherVariables))
lines.extend(generate_node_hints())

output_path = Path(var_hinting_path).with_suffix('.pyi')
output_path.write_text('\n'.join(lines))

# Run formatter
subprocess.run(
    ['ruff', 'check', '--select', 'I', '--fix', str(output_path)],
    check=True,
)
subprocess.run(['ruff', 'format', str(output_path)], check=True)
