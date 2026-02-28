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
    _ignore_defaults: bool
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
            cls._ignore_defaults = False

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

    @property
    def ignore_defaults(self):
        return self._ignore_defaults

    @ignore_defaults.setter
    def ignore_defaults(self, flag: bool):
        self._ignore_defaults = flag

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

        regex_match = self._find_regex_match(key, self._defaults)
        if regex_match is not None and not self._ignore_defaults:
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

    @property
    def _all_values(self) -> dict[K, V]:
        if self._ignore_defaults:
            return self._user_values
        else:
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
        # Restore defaults
        cls._instance._ignore_defaults = False
        # Reset fallback and forced values
        cls._instance._fallback_value = None
        cls._instance._forced_value = None

    def __contains__(self, key: K) -> bool:
        # Check exact match first
        if key in self._all_values:
            return True
        # Check regex patterns
        if self._find_regex_match(key, self._user_values) is not None:
            return True
        if (
            not self._ignore_defaults
            and self._find_regex_match(key, self._defaults) is not None
        ):
            return True
        return False


class DefaultUnitsRegistry(BaseRegistry[str, str]):
    """
    Registry to store the units of different variable types
    """

    DEFAULTS = {
        # Thermodynamics
        'p': 'Pa',
        'p_.*': 'Pa',  # WARN: this could catch unwanted stuff
        'T': 'K',
        'T_critical': 'K',
        '.*_T_is': 'K',  # used by isentropic temperature
        '.*_[a-zA-Z]_red': 'dimensionless',  # reduced quantities
        'rhomass': 'kg / m**3',
        'rhomass_.*': 'kg / m**3',
        '.*hmass.*': 'J / kg',  # Includes delta_hmass
        '.*smass.*': 'J / (kg * K)',  # Includes delta_smass
        'umass': 'J / kg',
        'cpmass.*': 'J / (kg * K)',
        'cvmass.*': 'J / (kg * K)',
        'speed_sound': 'm/s',
        'viscosity': 'Pa * s',
        # Others
        '.*massflow': 'kg / s',
        'mach.*': 'dimensionless',
        'mermach': 'dimensionless',
        'relmach.*': 'dimensionless',
        'reactDegree.*': 'dimensionless',
        'swllCap': 'dimensionless',
        'gamma_pv': 'dimensionless',
        # Dimensionless REGEX
        'eta_[s-t]{2}': 'dimensionless',  # eta_tt, eta_ts
        '.*Coeff': 'dimensionless',
        '.*Efficiency': 'dimensionless',
        '.*Func': 'dimensionless',
        '.*Ratio.*': 'dimensionless',
        'Cf_.*': 'dimensionless',  # Friction factor
        'Cd_.*': 'dimensionless',  # Friction factor
        # Kinematics
        'V': 'm/s',
        'Vt': 'm/s',
        'Vm': 'm/s',
        'W': 'm/s',
        'W_.*': 'm/s',
        'Wt': 'm/s',
        'Wm': 'm/s',
        'V[mt]?_.*': 'm/s',  # Vm_something, Vt_something, V_something
        'W[mt]?_.*': 'm/s',
        'U': 'm/s',
        'omega': 'rad/s',
        'beta': 'rad',
        'beta_.*': 'rad',
        'alpha': 'rad',
        'deflection': 'rad',
        # Geometry
        '.*_angle': 'rad',
        'meridional_angle': 'rad',
        'height': 'm',
        'hh': 'm',
        'rr': 'm',
        'rr_.*': 'm',
        'tip_clearance': 'm',
        'abs_roughness': 'm',
        '.*area': 'm**2',
        # Blade parameters
        'chord.*': 'm',
        'camb_len': 'm',
        'pitch': 'm',
        'throat': 'm',
        'stagger': 'rad',
        'metal_angle': 'rad',
        'metal_angle_.*': 'rad',
        '.*solidity': 'dimensionless',
        'slip_factor': 'dimensionless',
        'num_.*': 'dimensionless',
        '.*_thick.*': 'meters',
        'thick_by_pitch': 'dimensionless',
        '.*_by_.*': 'dimensionless',
    }


class GuessRegistry(BaseRegistry[str, float]):
    """
    Registry to store guesses based on variable type, in SI units
    """

    DEFAULTS = {
        # THERMODYNAMICS
        'p': 5e5,
        'p_base': 3e5,
        '.*p_red': 1.5,
        'hmass': 6e5,
        '.*hmass_is': 6e5,
        'umass': 5e5,
        'rhomass': 2.0,
        'T': 800.0,
        '.*T_red': 1.5,
        '.*T_is': 300.0,
        'smass': 1e4,
        'delta_smass_.*': 10.0,
        'eta_tt': 0.9,
        'gamma_pv': 1.4,
        # Main kinematics - VERY SENSITIVE
        'V': 1e-2,
        'Vm': 1e-2,
        'Vt': 1e-2,
        'W': 1e-2,
        'Wm': 1e-2,
        'Wt': 1e-2,
        'U': 1e-2,
        'omega': 1e-2,
        'alpha': 1e-2,
        'beta': 1e-2,
        # Secondary kinematics
        'VmRatio': 1.0,
        'V[mt]?_.*': 0.01,  # Vm_something, Vt_something, V_something
        'W[mt]?_.*': 0.01,
        'beta_.*': -0.3,
        'deflection': 1.0,
        'mach': 0.3,
        'mermach': 0.3,
        'relmach.*': 0.3,
        'dev_angle': 0.01,
        # OTHERS
        'massflow': 20.0,
        'flowCoeff': 0.8,
        'cum_massflow': 20.0,
        'ch_massflow': 1.0,
        # GEOMETRY
        'hubtipRatio': 0.5,
        'area': 0.1,
        'eff_area': 0.1,
        'cum_area': 0.3,
        'meridional_angle': 0.1,
        'metal_angle': -0.3,
        'solidity': 1.0,
        'hh': 0.1,
        'height': 0.1,
        'flare_angle': 0.1,
        'heightRatio': 1.0,
        'radiusRatio': 1.0,
        'tip_clearance': 0.001,
        'rr': 0.1,
        'rr_hub': 0.1,
        'rr_midspan': 0.11,
        'rr_tip': 0.12,
        'rrRatio': 0.4,
        'aspRatio': 2.0,
        'chord': 0.1,
        'chord_ax': 0.1,
        'camb_len': 0.1,
        'pitch': 0.1,
        'bld_thick': 0.005,
        'mom_thick': 1e-5,
        'disp_thick.*': 2e-5,
        'num_blades.*': 20.0,
        'num_splitters': 20.0,
        'throat': 0.1,
        'stagger': 0.1,
    }


ureg = UnitRegistry()

SCALING_FACTORS: dict[str, float] = {
    'm / s': 100.0,
    'Pa': 5e5,
    'N': 100.0,
    'N / m': 1000.0,
    'K': 500.0,
    'J / kg': 5e5,
    'J / kg / K': 1e3,  # Entropy
    'J / kg / m': 5e5,  # Radial equilibrium
    'kg / s': 5,
    'meter * rad / s': 1e2,  # Tangential velocity
    'Pa * s': 1e-5,  # Dynamic viscosity
    'rad / s': 1e3,  # Omega
    'rad': 0.5,
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

    def get(self, key: str):
        units = self._base_units_from_key(key)
        return super().get(units)


class ScalarsRegistry(BaseRegistry[str, int]):
    """
    Quantities that should be always be treated as scalars,

    Note:
    -----
    This may not always be necessary as the system may be
    well defined with some arguments that act as scalars
    but are just constant-valued vectors (e.g. cumulative
    massflow).

    Instead, if the problem is being solved FOR one of these variables,
    having them appear in the system at only at one
    spanwise station is problematic.

    For example, kin_omega0 (node 0) is unknown, but it appears
    only as kin_omega0[0] e.g. in SpeedLinker, because physically
    the user wants to make it constant. The root problem is apparently
    well defined (n equations = n unknowns), but the other `spurious`
    components of omega DO NOT appear in the system, meaning the
    rootfinder will always fail due to a rank deficiency in the Jacobian.
    """

    # The values do nothing, this acts as a list but I wanted to reuse
    # the base registry class without duplication
    # Could do a dual mode for the registry for clarity
    DEFAULTS = {
        'omega': -1,
        '.*_midspan': -1,
        '.*_hub': -1,
        '.*_tip': -1,
        'height': -1,
        'flare_angle': -1,
        'heightRatio': -1,
        'radiusRatio': -1,
        'hubtipRatio': -1,
        'aspRatio': -1,
        'cum_.*': -1,
        'meridional_angle': -1,
        'num_blades': -1,
        '.*AreaAve': -1,
        # Coefficients
        'flowCoeff': -1,
        'workCoeff': -1,
        'ts_loadCoeff': -1,
        'reactDegree.*': -1,
    }


# This registry is not actively used anywhere for now
class VariableBoundsRegistry(
    BaseRegistry[
        str,
        tuple[float, float],
    ]
):
    DEFAULTS = {
        # *** THERMODYNAMICS
        'p': (1e4, 150e5),
        'rhomass': (1e-3, 800.0),
        'T': (80.0, 1800.0),
        # *** KINEMATICS
        'V': (0.1, 600.0),
        'Vm': (0.1, 600.0),
        'Vt': (-600.0, 600.0),
        'W': (0.1, 600.0),
        'Wm': (0.1, 600.0),
        'Wt': (-600.0, 600.0),
        'U': (-600.0, 600.0),
        'omega': (-15000.0, 15000.0),
        'beta': (-1.45, 1.45),
        'alpha': (-1.45, 1.45),
        'metal_angle': (-1.45, 1.45),
        '.*area': (0.0, 2.0),
        'rr': (1e-4, 3.0),
        'rr_.*': (1e-4, 3.0),
        'hh': (1e-8, 1.0),
        'height': (1e-5, 3.0),
        # *** OTHERS
        '.*solidity': (0.05, 10.0),
        '.*massflow': (1e-4, 5e4),
        'slip_factor': (0.01, 0.99),
    }


if __name__ == '__main__':
    # Example: Test regex matching
    reg = DefaultUnitsRegistry()

    # Add a regex pattern to match all variables containing 'hmass'
    reg['.*hmass.*'] = 'J / kg'

    # Test exact match (should still work)
    print(f'hmass: {reg["hmass"]}')  # Exact match from DEFAULTS

    # Test regex match
    print(f'tot_hmass: {reg["tot_hmass"]}')  # Should match .*hmass.* pattern
    print(
        f'delta_hmass_custom: {reg["delta_hmass_custom"]}'
    )  # Should match .*hmass.* pattern

    # Test contains
    print(f"'custom_hmass_var' in reg: {'custom_hmass_var' in reg}")  # Should be True
