"""
Simple module to time functions and methods in classes
"""

import timeit
from functools import wraps
import types


def function_timer(custom_str=''):
    def decorator(func):
        def wrapper(*args, **kwargs):
            time0 = timeit.default_timer()
            result = func(*args, **kwargs)
            time1 = timeit.default_timer()

            print(f'{custom_str} {func.__name__} took {time1 - time0:.2e} seconds')
            return result

        return wrapper

    return decorator


def class_timer(custom_str=''):
    """Class decorator that times all methods in a class.

    Args:
        custom_str (str): Optional string to prepend to timing output

    Example:
    --------

        >>> @class_timer("MyClass timing:")
        >>> class MyClass:
        >>>     def method1(self):
                ...
    """

    def timer_wrapper(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            time0 = timeit.default_timer()
            result = f(*args, **kwargs)
            time1 = timeit.default_timer()
            print(f'{custom_str} {f.__name__} took {time1 - time0:.2e} seconds')
            return result

        return wrapper

    def decorate(cls):
        for attr_name, attr_value in cls.__dict__.items():
            # Skip special methods and non-callable attributes
            if attr_name.startswith('__'):
                continue

            # Handle different method types
            if isinstance(attr_value, types.FunctionType):
                # Regular instance method
                setattr(cls, attr_name, timer_wrapper(attr_value))
            elif isinstance(attr_value, classmethod):
                # Class method
                original_method = attr_value.__get__(None, cls).__func__
                setattr(cls, attr_name, classmethod(timer_wrapper(original_method)))
            elif isinstance(attr_value, staticmethod):
                # Static method
                original_method = attr_value.__get__(None, cls)
                setattr(cls, attr_name, staticmethod(timer_wrapper(original_method)))

        return cls

    return decorate
