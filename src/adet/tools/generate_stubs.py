from typing import Type, Any
from pathlib import Path
import subprocess
import importlib.util
import inspect

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
    GenericVariables,
    OtherHints,
    NodeHints,
    NodeStates,
    CustomHints,
)


def get_annotation_string(param: inspect.Parameter):
    if hasattr(param.annotation, '__name__'):
        return param.annotation.__name__
    else:
        return '|'.join(
            [a.__name__.replace('Type', '') for a in param.annotation.__args__]
        )


def generate_init_signature(class_to_read: Type[Any]):
    sig = inspect.signature(class_to_read.__init__)
    init_params = []
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        annotation_str = get_annotation_string(param)
        params_str = f'{name}: {annotation_str}'
        init_params.append(params_str)

    params_str = ', '.join(init_params)

    return f'    def __init__(self, {params_str}): ...'


def generate_dataclass_ppties(class_to_read):
    sig = inspect.signature(class_to_read.__init__)
    properties = []
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        annotation_str = get_annotation_string(param)

        properties.append('    @property')
        properties.append(f'    def {name}(self) -> {annotation_str}: ...')

    return properties


def generate_hint_class(
    hint_class: Type[VariableHints],
    enum_class: Type[ThermoVariables | GenericVariables],
):
    class_lines = []
    class_lines.append(f'class {hint_class.__name__}({VariableHints.__name__}):')

    # Read actual __init__ signature
    sig = inspect.signature(hint_class.__init__)
    init_params = []
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        params_str = f'{name}: {param.annotation.__name__}'
        init_params.append(params_str)

    class_lines.append(generate_init_signature(hint_class))

    # Generate properties from enum members
    for var in enum_class:
        class_lines.append(
            f'    {var.name} = Annotated[MX | Quantity, {VarSpec.__name__}]'
        )
    return class_lines


def generate_node_hints():
    class_lines = [
        f'class {NodeHints.__name__}({OtherHints.__name__}):',
        '    def __init__(self, index: int): ...',
    ]

    # Add ThermoHints properties (stc, tot, rlt)
    for prop_name in NodeStates:
        class_lines.extend(
            [
                '    @property',
                f'    def {prop_name.value}(self) -> {ThermoHints.__name__}: ...',
            ]
        )
    class_lines.extend(
        [
            '    @property',
            f'    def cust(self) -> {CustomHints.__name__}: ...',
        ]
    )

    return class_lines


if __name__ == '__main__':
    # Imports
    lines = [
        'from enum import Enum',
        'from casadi import MX',
        'from pint import Quantity',
        'from typing import Annotated',
    ]

    # Generate VarSpec class with actual properties
    lines.append(f'class {VarSpec.__name__}:')
    lines.append(generate_init_signature(VarSpec))
    lines.extend(generate_dataclass_ppties(VarSpec))

    # Simple classes
    lines.extend(
        [
            f'class {VariableHints.__name__}: ...',
            f'class {GenericVariables.__name__}(Enum): ...',
            f'class {ThermoVariables.__name__}(Enum): ...',
            f'class {CustomHints.__name__}(Enum): ...',
            f'class {NodeStates.__name__}(Enum): ...',
        ],
    )

    lines.extend(generate_hint_class(ThermoHints, ThermoVariables))
    lines.extend(generate_hint_class(OtherHints, GenericVariables))
    lines.extend(generate_node_hints())

    output_path = Path(var_hinting_path).with_suffix('.pyi')
    output_path.write_text('\n'.join(lines))

    # Run formatter
    subprocess.run(
        ['ruff', 'check', '--select', 'I', '--fix', str(output_path)],
        check=True,
    )
    subprocess.run(['ruff', 'format', str(output_path)], check=True)
