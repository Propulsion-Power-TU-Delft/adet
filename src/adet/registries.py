"""
Implement registries with a singleton pattern
"""

import logging
from pint import UnitRegistry
from typing import Generic, TypeVar, Mapping

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

            # Initialize
            cls._user_values = {}
            cls._fallback_value = None
            cls._forced_value = None

        return cls._instance

    def set_fallback_value(self, value: V):
        self._fallback_value = value

    def set_forced_value(self, value: V):
        self._forced_value = value

    def set(self, key: K, value: V) -> None:
        self._user_values[key] = value

    def __setitem__(self, key, value):
        # These are intentionally left untyped
        self.set(key, value)

    def get(self, key: K) -> V:
        # Return the forced value if assigned
        if self._forced_value:
            return self._forced_value

        # Raise if there is no key
        if key not in self._all_values:
            if self._fallback_value:
                return self._fallback_value

            class_name = self.__class__.__name__
            raise KeyError(
                f'Missing key `{key}` for {class_name}, '
                f"add it using: `{class_name}()['{key}'] = <value>`"
            )

        # User values take precedence over defaults
        if key in self._user_values:
            return self._user_values[key]

        return self._defaults[key]

    def __getitem__(self, key):
        # These are intentionally left untyped
        return self.get(key)

    @property
    def _all_values(self) -> dict[K, V]:
        return {**self._defaults, **self._user_values}

    def from_dict(self, input: dict[K, V]):
        for k, v in input.items():
            self.set(k, v)

    def clear(self) -> None:
        """Remove user values"""
        self._user_values.clear()

    @classmethod
    def reset(cls) -> None:
        if cls._instance is None:
            raise AttributeError('Resetting a unitialized registry is not allowed')

        # Remove custom values
        cls._instance.clear()
        # Reset fallback and forced values
        cls._instance._fallback_value = None
        cls._instance._forced_value = None

    def __contains__(self, key: K) -> bool:
        return key in self._all_values


class DefaultUnitsRegistry(BaseRegistry[str, str]):
    """
    Registry to store the units of different variable types
    """

    DEFAULTS = {
        # Thermodynamics
        'p': 'Pa',
        'p_ref': 'Pa',
        'T': 'K',
        'T_ref': 'K',
        'rhomass': 'kg / m**3',
        'hmass': 'J / kg',
        'umass': 'J / kg',
        'smass': 'J / (kg * K)',
        'cpmass': 'J / (kg * K)',
        'cpmassid': 'J / (kg * K)',
        'cvmass': 'J / (kg * K)',
        'cvmassid': 'J / (kg * K)',
        'speed_sound': 'm/s',
        'Ma': 'dimensionless',
        # Others
        'massflow': 'kg / s',
        'cum_massflow': 'kg / s',
        # Kinematics
        'V': 'm/s',
        'Vt': 'm/s',
        'Vm': 'm/s',
        'W': 'm/s',
        'Wt': 'm/s',
        'Wm': 'm/s',
        'U': 'm/s',
        'omega': 'rad/s',
        'beta': 'rad',
        'alpha': 'rad',
        # Meridional Geometry
        'meridional_angle': 'rad',
        'height': 'm',
        'hh': 'm',
        'rr': 'm',
        'rmid': 'm',
        'area': 'm**2',
        # Blade parameters
        'chord': 'm',
        'chord_ax': 'm',
        'camb_len': 'm',
        'pitch': 'm',
        'stagger': 'rad',
    }


class GuessRegistry(BaseRegistry[str, float]):
    """
    Registry to store guesses based on variable type
    """

    DEFAULTS = {
        # THERMODYNAMICS
        'p': 5e5,
        'hmass': 6e5,
        'umass': 5e5,
        'rhomass': 2.0,
        'T': 800.0,
        'smass': 1e4,
        # KINEMATICS
        'V': 300.0,
        'Vm': 300.0,
        'Vt': 10.0,
        'W': 300.0,
        'Wm': 300.0,
        'Wt': 10.0,
        'U': 10.0,
        'omega': 10.0,
        'alpha': 0.3,
        'beta': 0.3,
        # OTHERS
        'massflow': 20.0,
        # GEOMETRY
        'area': 0.1,
        'meridional_angle': 0.1,
        'hh': 0.1,
        'height': 0.1,
        'rr': 0.1,
        'rmid': 0.1,
        'chord': 0.1,
        'chord_ax': 0.1,
        'pitch': 0.1,
        'stagger': 0.1,
    }


ureg = UnitRegistry()

SCALING_FACTORS: dict[str, float] = {
    'm / s': 100.0,
    'Pa': 5e5,
    'K': 500.0,
    'J / kg': 5e5,
    'J / (kg * K)': 1e3,
    'kg / s': 5e1,
    'rad / s': 1e3,
    'rad': 0.5,
    'm': 1e-1,
    'm**2': 1e-2,
    'kg / m**3': 2.0,
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

    def get(self, key: str, fallback=None):
        units = self._base_units_from_key(key)
        return super().get(units)


# This registry is not actively used anywhere for now
class VariableBoundsRegistry(
    BaseRegistry[
        str,
        tuple[float, float],
    ]
):
    DEFAULTS = {
        # THERMODYNAMICS
        'p': (1e4, 50e5),
        'hmass': (1e3, 1e7),
        'umass': (1e3, 1e7),
        'rhomass': (1e-3, 50.0),
        'T': (40.0, 1800.0),
        'smass': (-1e5, 1e5),
        # KINEMATICS
        'V': (10.0, 500.0),
        'Vm': (10.0, 500.0),
        'Vt': (-500.0, 500.0),
        'W': (10.0, 500.0),
        'Wm': (10.0, 500.0),
        'Wt': (-500.0, 500.0),
        'U': (-500.0, 500.0),
        'omega': (-10000.0, 10000.0),
        'alpha': (-1.5, 1.5),
        'beta': (-1.5, 1.5),
        'area': (0.0, 1.0),
        'rr': (1e-4, 1.0),
        'rmid': (1e-4, 1.0),
        'hh': (1e-5, 1.0),
        'height': (1e-5, 1.0),
        # OTHERS
        'massflow': (1e-3, 100.0),
    }


if __name__ == '__main__':
    reg = DefaultUnitsRegistry()
    reg['test']
