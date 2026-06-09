"""
Functions that act as context managers
"""

import os
import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Callable


@contextmanager
def override_function_globals(func, **overrides):
    """
    Examples
    --------
    >>> def my_function(x):
    >>>    return np.tan(x)
    >>> with override_function_globals(my_function, np=sp):
    >>>     print(my_function(sp.symbols('x')))
    tan(x)
    """
    original_globals = func.__globals__.copy()
    func.__globals__.update(overrides)
    try:
        yield
    finally:
        func.__globals__.clear()
        func.__globals__.update(original_globals)


def override_operators(
    func: Callable,
    module_to_override: str,
    source_module: ModuleType,
) -> Callable:
    """
    Override the module with specified name with the module
    provided
    """
    overrides = {}
    for key, value in func.__globals__.items():
        if hasattr(value, '__name__') and value.__name__ == module_to_override:
            overrides[key] = source_module
            break

    # No overrides found -> Return the original function
    if not overrides:
        return func

    def overridden_func(*args, **kwargs):
        with override_function_globals(func, **overrides):
            return func(*args, **kwargs)

    return overridden_func


@contextmanager
def output_suppression():
    """Suppress all stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@contextmanager
def dummy_context():
    """
    Does nothing, useful for if expressions.

    Example:
    --------
    ```
    if use_context:
        cont_manager = some_manager
    else:
        cont_manager = dummy_context

    with cont_manager():
        ...
    ```
    """

    try:
        yield
    finally:
        pass
