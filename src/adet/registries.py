"""Implement registries with a singleton pattern"""

import logging
import re
from typing import Generic, Mapping, TypeVar

from pint import UnitRegistry
from pint.facets.plain import PlainUnit

logger = logging.getLogger(__name__)

K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class BaseRegistry(Generic[K, V]):
    """
    Generic registry object, singleton pattern to store
    defaults across objects
    """

    DEFAULTS: Mapping[K, V]

    _defaults: dict[K, V]
    _user_values: dict[K, V]
    _fallback_value: V | None
    _forced_value: V | None
    ignore_defaults: bool

    def __init_subclass__(cls) -> None:
        # Check that the subclass defines defaults
        if not hasattr(cls, 'DEFAULTS'):
            raise KeyError('Missing `DEFAULTS` class variable')
        # Create the global instance
        cls._instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._defaults = dict(cls.DEFAULTS)
            cls.ignore_defaults = False

            # Initialize
            cls._user_values = {}
            cls._fallback_value = None
            cls._forced_value = None

        return cls._instance

    def set_fallback_value(self, value: V):
        """Set a value that is used in the event of a key miss"""
        self._fallback_value = value

    def set_forced_value(self, value: V):
        """Set a value that overriddes all defaults and user-defined values"""
        self._forced_value = value

    def set(self, key: K, value: V) -> None:
        self._user_values[key] = value

    def __setitem__(self, key, value):
        # These are intentionally left untyped
        self.set(key, value)

    def _find_regex_match(self, key: K, values: dict[K, V]) -> V | None:
        """Find a value by matching key against regex patterns in the registry."""
        if not isinstance(key, str):
            return None

        for pattern, value in values.items():
            if not isinstance(pattern, str):
                continue
            try:
                if re.fullmatch(pattern, key):
                    return value
            except re.error:
                # Not a valid regex pattern, skip
                continue
        return None

    def get(self, key: K) -> V:
        # Return the forced value if assigned
        if self._forced_value is not None:
            return self._forced_value

        # Try exact match first
        if key in self._all_values:
            # User values take precedence over defaults
            if key in self._user_values:
                return self._user_values[key]
            return self._defaults[key]

        # Try regex matching: user patterns first, then defaults
        regex_match = self._find_regex_match(key, self._user_values)
        if regex_match is not None:
            return regex_match

        regex_match = self._find_regex_match(key, self.defaults)
        if regex_match is not None:
            return regex_match

        # No match found - check fallback or raise error
        if self._fallback_value is not None:
            return self._fallback_value

        class_name = self.__class__.__name__
        raise KeyError(
            f'Missing key `{key}` for {class_name}, '
            f"add it using: `{class_name}()['{key}'] = <value>`"
        )

    def __getitem__(self, key):
        # These are intentionally left untyped
        return self.get(key)

    def __contains__(self, key: K) -> bool:
        # Check exact match first
        if key in self._all_values:
            return True
        # Check regex patterns
        if self._find_regex_match(key, self._user_values) is not None:
            return True
        if self._find_regex_match(key, self.defaults) is not None:
            return True
        return False

    @property
    def _all_values(self) -> dict[K, V]:
        return {**self.defaults, **self._user_values}

    @property
    def defaults(self) -> dict[K, V]:
        if self.ignore_defaults:
            return {}
        else:
            return self._defaults

    def from_dict(self, input: dict[K, V]):
        for k, v in input.items():
            self.set(k, v)

    def clear(self) -> None:
        """Clear user values"""
        self._user_values.clear()

    @classmethod
    def reset(cls) -> None:
        if cls._instance is None:
            raise RuntimeError('Resetting a unitialized registry is not allowed')

        # Remove custom values
        cls._instance.clear()
        # Restore defaults
        cls._instance.ignore_defaults = False
        # Reset fallback and forced values
        cls._instance._fallback_value = None
        cls._instance._forced_value = None


ureg = UnitRegistry()

SCALING_FACTORS: dict[str, float] = {
    'm / s': 100.0,
    'Pa': 5e5,
    'K': 500.0,
    'N': 100.0,
    'N / m': 1000.0,
    'J / kg': 5e5,
    'J / kg / K': 1e3,  # Entropy
    'J / kg / m': 5e5,  # Radial equilibrium
    '1 / s': 1e3,  # Forced vortex
    'kg / s': 5,
    'meter * rad / s': 1e2,  # Tangential velocity
    'kg / m**2 / s': 1e3,  # pass
    'Pa * s': 1e-5,  # Dynamic viscosity
    'rad / s': 1e3,  # Omega
    'rad': 1.0,
    'm': 0.1,
    'm**2': 1e-2,
    'm**2 / s': 10.0,
    'kg / m**3': 2.0,  # Densities
    'dimensionless': 1.0,
}
# Convert to standard string representation
DEFAULT_SCALES = {ureg(k).to_base_units().units: v for k, v in SCALING_FACTORS.items()}


# Scaling registry needs some overloading for passing
# between strings and units objects
class ScalingRegistry(BaseRegistry[PlainUnit, float]):
    """
    Registry for scaling factors based on units
    """

    DEFAULTS = DEFAULT_SCALES.copy()

    def _base_units_from_key(self, key: str):
        return ureg(key).to_base_units().units

    def set(self, key: str, value: float):
        units = self._base_units_from_key(key)
        return super().set(units, value)

    def get(self, key: str) -> float:
        units = self._base_units_from_key(key)
        return super().get(units)
