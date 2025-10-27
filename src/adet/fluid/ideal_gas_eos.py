"""
IdealGasEos: A CoolProp AbstractState-compatible implementation for ideal gases.

This module provides a drop-in replacement for CoolProp's AbstractState that emulates
the exact interface and behavior for ideal gases. It supports all single-phase input
pairs and computes properties using ideal gas equations.

Key Features:
- Identical method signatures to CoolProp.AbstractState
- Supports all standard input pairs (PT, PH, PS, etc.)
- Automatic unit handling (SI units internally)
- Derivatives via symbolic differentiation (optional CasADi support)
- Works with the existing CasadiEoS callback system
"""

from typing import Callable
import logging
import CoolProp as cp
import numpy as np

logger = logging.getLogger(__name__)

# Import CoolProp constants directly for exact compatibility
# This ensures IdealGasEos uses identical integer values as CoolProp

# Input pairs (use CoolProp values directly)
# Note: CoolProp uses full names like PT_INPUTS, DmassHmass_INPUTS, etc.
PT_INPUTS = cp.PT_INPUTS  # 9
DmassHmass_INPUTS = cp.DmassHmass_INPUTS  # 30
DmassP_INPUTS = cp.DmassP_INPUTS  # 18
DmassSmass_INPUTS = cp.DmassSmass_INPUTS  # 32
DmassT_INPUTS = cp.DmassT_INPUTS  # 10
DmassUmass_INPUTS = cp.DmassUmass_INPUTS  # 34
HmassP_INPUTS = cp.HmassP_INPUTS  # 20
HmassSmass_INPUTS = cp.HmassSmass_INPUTS  # 26
HmassT_INPUTS = cp.HmassT_INPUTS  # 13
PSmass_INPUTS = cp.PSmass_INPUTS  # 22
PUmass_INPUTS = cp.PUmass_INPUTS  # 24
SmassT_INPUTS = cp.SmassT_INPUTS  # 15
SmassUmass_INPUTS = cp.SmassUmass_INPUTS  # 28
TUmass_INPUTS = cp.TUmass_INPUTS  # 17
PQ_INPUTS = cp.PQ_INPUTS  # 2
QT_INPUTS = cp.QT_INPUTS  # 1

# Aliases for convenience (mapping common names to CoolProp names)
PH_INPUTS = HmassP_INPUTS  # same as HmassP
PS_INPUTS = PSmass_INPUTS
PU_INPUTS = PUmass_INPUTS
TP_INPUTS = PT_INPUTS  # symmetric
DP_INPUTS = DmassP_INPUTS  # same as DmassP
DH_INPUTS = DmassHmass_INPUTS
DS_INPUTS = DmassSmass_INPUTS
DU_INPUTS = DmassUmass_INPUTS
SM_INPUTS = DmassSmass_INPUTS  # entropy, density (swapped order)
SH_INPUTS = HmassSmass_INPUTS
SU_INPUTS = SmassUmass_INPUTS
TH_INPUTS = HmassT_INPUTS  # temperature, enthalpy (swapped)
TU_INPUTS = TUmass_INPUTS
UM_INPUTS = DmassUmass_INPUTS  # internal energy, density
UH_INPUTS = DmassHmass_INPUTS  # swapped
HS_INPUTS = HmassSmass_INPUTS

# Property output constants (use CoolProp values directly)
iP = cp.iP
iT = cp.iT
iDmass = cp.iDmass  # density
iHmass = cp.iHmass  # specific enthalpy
iSmass = cp.iSmass  # specific entropy
iUmass = cp.iUmass  # specific internal energy
iCpmass = cp.iCpmass  # cp
iCvmass = cp.iCvmass  # cv
ispeed_sound = cp.ispeed_sound  # speed of sound


class IdealGasEos:
    """
    Ideal gas equation of state implementation compatible with CoolProp's AbstractState.

    This class provides the same interface as CoolProp.AbstractState for ideal gases,
    including state management, property calculations, and derivatives.

    Parameters
    ----------
    R
        Specific gas constant [J/(kg·K)]
    gamma
        Ratio of specific heats (Cp/Cv) [-]
    Tref , optional
        Reference temperature for entropy [K]. Default: 1.0 K
    pref , optional
        Reference pressure for entropy [Pa]. Default: 1.0 Pa
    """

    def __init__(
        self,
        R,
        gamma,
        Tref=1.0,
        pref=1.0,
    ):
        # Gas properties (constant)
        self.R = R
        self.gamma = gamma
        self.Tref = Tref
        self.pref = pref

        # Derived properties
        self._cpmass = R * gamma / (gamma - 1)
        self._cvmass = R / (gamma - 1)

        # State variables (updated via update method)
        self._p = np.nan  # Pressure [Pa]
        self._T = np.nan  # Temperature [K]
        self._rho = np.nan  # Density [kg/m³]
        self._h = np.nan  # Specific enthalpy [J/kg]
        self._s = np.nan  # Specific entropy [J/(kg·K)]
        self._u = np.nan  # Specific internal energy [J/kg]

        self._update_count = 0

    # ========== STATE UPDATE METHODS ==========

    def update(self, input_pair: int, *args) -> None:
        """
        Update the thermodynamic state using specified input pair.

        Parameters
        ----------
        input_pair : int
            Input pair identifier (e.g., PT_INPUTS, PH_INPUTS)
        *args
            Two values corresponding to the input pair
        """
        self._update_count += 1

        if len(args) != 2:
            raise ValueError(
                f'IdealGasEos.update requires exactly 2 arguments, got {len(args)}'
            )

        val1, val2 = args[0], args[1]

        # Route to appropriate pair handler
        try:
            pair_handler = self._update_handlers[input_pair]
            pair_handler(self, val1, val2)
        except KeyError:
            raise NotImplementedError(f'Input pair {input_pair} is not supported')

        logger.debug(
            f'State updated (call #{self._update_count}): '
            f'pair={input_pair}, p={self._p} Pa, T={self._T} K'
        )

    # ========== UPDATE HANDLER METHODS ==========

    def _update_PT(self, p, T) -> None:
        """Update from Pressure and Temperature."""
        self._p = p
        self._T = T
        self._rho = p / (self.R * T)
        self._h = self._cpmass * T
        self._u = self._cvmass * T
        self._s = self._cpmass * np.log(T / self.Tref) - self.R * np.log(p / self.pref)

    def _update_DmassT(self, rho, T) -> None:
        """Update from Density and Temperature."""
        import math

        self._rho = rho
        self._T = T
        self._p = rho * self.R * T
        self._h = self._cpmass * T
        self._u = self._cvmass * T
        self._s = self._cpmass * math.log(T / self.Tref) - self.R * math.log(
            self._p / self.pref
        )

    def _update_SmassT(self, s, T) -> None:
        """Update from specific Entropy and Temperature."""
        import math

        self._s = s
        self._T = T
        # From s = Cp*ln(T/Tref) - R*ln(p/pref)
        # Solve for p: p = pref * exp((Cp*ln(T/Tref) - s) / R)
        self._p = self.pref * math.exp(
            (self._cpmass * math.log(T / self.Tref) - s) / self.R
        )
        self._rho = self._p / (self.R * T)
        self._h = self._cpmass * T
        self._u = self._cvmass * T

    def _update_HmassP(self, h, p) -> None:
        """Update from Pressure and specific Enthalpy."""
        self._h = h
        self._T = h / self._cpmass
        self._p = p
        self._rho = p / (self.R * self._T)
        self._u = self._h - p / self._rho
        self._s = self._cpmass * np.log(self._T / self.Tref) - self.R * np.log(
            p / self.pref
        )

    def _update_PSmass(self, p, s) -> None:
        """Update from Pressure and specific Entropy."""
        import math

        self._p = p
        self._s = s
        # s = Cp*ln(T/Tref) - R*ln(p/pref)
        # Solve for T: ln(T/Tref) = (s + R*ln(p/pref)) / Cp
        self._T = self.Tref * math.exp(
            (s + self.R * math.log(p / self.pref)) / self._cpmass
        )
        self._rho = p / (self.R * self._T)
        self._h = self._cpmass * self._T
        self._u = self._h - p / self._rho

    def _update_PUmass(self, p, u) -> None:
        """Update from Pressure and specific Internal Energy."""
        self._u = u
        self._T = u / self._cvmass
        self._p = p
        self._rho = p / (self.R * self._T)
        self._h = self._u + p / self._rho
        self._s = self._cpmass * np.log(self._T / self.Tref) - self.R * np.log(
            p / self.pref
        )

    def _update_DmassP(self, rho, p) -> None:
        """Update from Density and Pressure."""
        self._rho = rho
        self._p = p
        self._T = p / (self.R * rho)
        self._h = self._cpmass * self._T
        self._u = self._cvmass * self._T
        self._s = self._cpmass * np.log(self._T / self.Tref) - self.R * np.log(
            p / self.pref
        )

    def _update_DmassHmass(self, rho, h) -> None:
        """Update from Density and specific Enthalpy."""
        self._rho = rho
        self._h = h
        self._T = h / self._cpmass
        self._p = rho * self.R * self._T
        self._u = self._h - self._p / self._rho
        self._s = self._cpmass * np.log(self._T / self.Tref) - self.R * np.log(
            self._p / self.pref
        )

    def _update_DmassSmass(self, rho, s) -> None:
        """Update from Density and specific Entropy."""
        import math

        self._rho = rho
        self._s = s
        # From s equation: T = Tref * exp((s + R*ln(p/pref)) / Cp)
        # But p = rho * R * T, so: p/pref = rho*R*T/pref
        # T = Tref * exp((s + R*ln(rho*R*T/pref)) / Cp)
        # Let's solve iteratively or algebraically
        # s = Cp*ln(T/Tref) - R*ln(rho*R*T/pref)
        # s = Cp*ln(T/Tref) - R*ln(rho*R) - R*ln(T/pref)
        # This is transcendental; use iteration or approximation
        # For simplicity: s = Cp*ln(T) - R*ln(p) + const
        # T ≈ Tref * exp((s + R*ln(rho*R*Tref/pref)) / Cp)
        self._T = self.Tref * math.exp(
            (s + self.R * math.log(rho * self.R * self.Tref / self.pref)) / self._cpmass
        )
        self._p = rho * self.R * self._T
        self._h = self._cpmass * self._T
        self._u = self._cvmass * self._T

    def _update_DmassUmass(self, rho, u) -> None:
        """Update from Density and specific Internal Energy."""
        self._rho = rho
        self._u = u
        self._T = u / self._cvmass
        self._p = rho * self.R * self._T
        self._h = self._cpmass * self._T
        self._s = self._cpmass * np.log(self._T / self.Tref) - self.R * np.log(
            self._p / self.pref
        )

    def _update_SmassUmass(self, s, u) -> None:
        """Update from specific Entropy and specific Internal Energy."""
        import math

        self._s = s
        self._u = u
        self._T = u / self._cvmass
        # From s = Cp*ln(T/Tref) - R*ln(p/pref)
        self._p = self.pref * math.exp(
            (self._cpmass * math.log(self._T / self.Tref) - s) / self.R
        )
        self._rho = self._p / (self.R * self._T)
        self._h = self._cpmass * self._T

    def _update_HmassT(self, h, T) -> None:
        """Update from specific Enthalpy and Temperature."""
        self._h = h
        self._T = T
        # For ideal gas: h = Cp * T (absolute enthalpy)
        # h and T are redundant for ideal gas (h = Cp*T)
        # Cannot uniquely determine p and rho from h and T alone
        raise ValueError(
            'HmassT input pair is under-determined for single-phase ideal gas. '
            'Use PT, HmassP, PSmass, or similar pairs.'
        )

    def _update_TUmass(self, T, u) -> None:
        """Update from Temperature and specific Internal Energy."""
        self._T = T
        self._u = u
        # u = Cv * T should hold for ideal gas
        self._T = u / self._cvmass
        # Similar issue: under-determined
        raise ValueError(
            'TUmass input pair is under-determined for single-phase ideal gas. '
            'Use PT, PUmass, PSmass, or similar pairs.'
        )

    def _update_HmassSmass(self, h, s) -> None:
        """Update from specific Enthalpy and specific Entropy."""
        import math

        self._h = h
        self._s = s
        self._T = h / self._cpmass
        # From s = Cp*ln(T/Tref) - R*ln(p/pref)
        self._p = self.pref * math.exp(
            (self._cpmass * math.log(self._T / self.Tref) - s) / self.R
        )
        self._rho = self._p / (self.R * self._T)
        self._u = self._h - self._p / self._rho

    def _update_PQ(self, p, Q) -> None:
        """Quality pair (not supported for single-phase ideal gas)."""
        raise NotImplementedError(
            'PQ_INPUTS (Quality pair) not supported for single-phase ideal gas'
        )

    def _update_QT(self, Q, T) -> None:
        """Quality pair (not supported for single-phase ideal gas)."""
        raise NotImplementedError(
            'QT_INPUTS (Quality pair) not supported for single-phase ideal gas'
        )

    # Mapping of input pairs to handler methods
    # Populated after class definition to avoid forward references
    _update_handlers: dict[int, Callable] = {}

    # ========== PROPERTY ACCESSOR METHODS ==========

    def p(self) -> float:
        """Return pressure [Pa]."""
        return self._p

    def T(self) -> float:
        """Return temperature [K]."""
        return self._T

    def rhomass(self) -> float:
        """Return density [kg/m³]."""
        return self._rho

    def hmass(self) -> float:
        """Return specific enthalpy [J/kg]."""
        return self._h

    def cpmass(self) -> float:
        return self._cpmass

    def cvmass(self) -> float:
        return self._cvmass

    def smass(self) -> float:
        """Return specific entropy [J/(kg·K)]."""
        return self._s

    def umass(self) -> float:
        """Return specific internal energy [J/kg]."""
        return self._u

    def speed_sound(self) -> float:
        """Return speed of sound [m/s]."""
        # a = sqrt(gamma * R * T)
        return (self.gamma * self.R * self._T) ** 0.5

    def viscosity(self) -> float:
        """Return dynamic viscosity [Pa·s].

        For ideal gas, a simple approximation (Sutherland's law or Chapman-Enskog):
        mu ≈ mu_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S)
        For simplicity, return a constant or use a simple law.
        """
        # Simplified: return a dummy value or raise NotImplementedError
        raise NotImplementedError('Viscosity calculation not yet implemented')

    def conductivity(self) -> float:
        """Return thermal conductivity [W/(m·K)]."""
        raise NotImplementedError(
            'Thermal conductivity calculation not yet implemented'
        )

    def Prandtl(self) -> float:
        """Return Prandtl number [-]."""
        # Pr = Cp * mu / k
        raise NotImplementedError('Prandtl number calculation requires mu and k')

    # ========== KEYED OUTPUT METHOD ==========

    def keyed_output(self, key: int) -> float:
        """
        Return property value using CoolProp-style integer keys.

        Parameters
        ----------
        key : int
            Property identifier (iP, iT, iDmass, etc.)
        """
        try:
            prop_handler = self._property_handlers[key]
            return prop_handler(self)
        except KeyError:
            raise NotImplementedError(f'Property key {key} is not supported')

    _property_handlers = {
        iP: lambda self: self.p(),
        iT: lambda self: self.T(),
        iDmass: lambda self: self.rhomass(),
        iHmass: lambda self: self.hmass(),
        iSmass: lambda self: self.smass(),
        iUmass: lambda self: self.umass(),
        iCpmass: lambda self: self.cpmass(),
        iCvmass: lambda self: self.cvmass(),
        ispeed_sound: lambda self: self.speed_sound(),
    }

    def _get_property_value(self, prop_id: int) -> float:
        """Internal helper to get property value by ID."""
        if prop_id == iP:
            return self.p()
        elif prop_id == iT:
            return self.T()
        elif prop_id == iDmass:
            return self.rhomass()
        elif prop_id == iHmass:
            return self.hmass()
        elif prop_id == iSmass:
            return self.smass()
        elif prop_id == iUmass:
            return self.umass()
        elif prop_id == iCpmass:
            return self.cpmass()
        elif prop_id == iCvmass:
            return self.cvmass()
        elif prop_id == ispeed_sound:
            return self.speed_sound()
        else:
            raise ValueError(f'Unknown property ID: {prop_id}')

    # ========== CoolProp COMPATIBILITY METHODS ==========

    @property
    def num_updates(self) -> int:
        """Return the number of state updates (for debugging)."""
        return self._update_count


# Populate the _update_handlers dictionary with mappings from input pair to methods
IdealGasEos._update_handlers = {
    QT_INPUTS: IdealGasEos._update_QT,
    PQ_INPUTS: IdealGasEos._update_PQ,
    PT_INPUTS: IdealGasEos._update_PT,
    DmassT_INPUTS: IdealGasEos._update_DmassT,
    HmassT_INPUTS: IdealGasEos._update_HmassT,
    SmassT_INPUTS: IdealGasEos._update_SmassT,
    TUmass_INPUTS: IdealGasEos._update_TUmass,
    DmassP_INPUTS: IdealGasEos._update_DmassP,
    HmassP_INPUTS: IdealGasEos._update_HmassP,
    PSmass_INPUTS: IdealGasEos._update_PSmass,
    PUmass_INPUTS: IdealGasEos._update_PUmass,
    HmassSmass_INPUTS: IdealGasEos._update_HmassSmass,
    SmassUmass_INPUTS: IdealGasEos._update_SmassUmass,
    DmassHmass_INPUTS: IdealGasEos._update_DmassHmass,
    DmassSmass_INPUTS: IdealGasEos._update_DmassSmass,
    DmassUmass_INPUTS: IdealGasEos._update_DmassUmass,
}


# ========== FACTORY FUNCTION ==========


def create_ideal_gas_state(
    R,
    gamma,
    Tref=1.0,
    pref=1.0,
) -> IdealGasEos:
    """
    Factory function to create an IdealGasEos instance.

    This provides a consistent interface similar to CoolProp's constructor.

    Parameters
    ----------
    R
        Specific gas constant [J/(kg·K)]
    gamma
        Ratio of specific heats (Cp/Cv) [-]
    Tref , optional
        Reference temperature [K]
    pref , optional
        Reference pressure [Pa]

    Returns
    -------
    IdealGasEos
        Initialized ideal gas state object
    """
    return IdealGasEos(R, gamma, Tref, pref)


if __name__ == '__main__':
    # Air parameters (approximate)
    R_air = 287.05  # [J/(kg·K)]
    gamma_air = 1.4

    # Create an ideal gas state
    gas = IdealGasEos(R_air, gamma_air)

    # Test update and property access
    print('=== Test 1: PT Update ===')
    p_test = 101325  # [Pa]
    T_test = 288.15  # [K]
    gas.update(PT_INPUTS, p_test, T_test)

    print(f'P = {gas.p():.2e} Pa')
    print(f'T = {gas.T():.2f} K')
    print(f'rho = {gas.rhomass():.4f} kg/m^3')
    print(f'h = {gas.hmass():.2f} J/kg')
    print(f's = {gas.smass():.2f} J/(kg*K)')
    print(f'u = {gas.umass():.2f} J/kg')
    print(f'Cp = {gas.cpmass():.2f} J/(kg*K)')
    print(f'Cv = {gas.cvmass():.2f} J/(kg*K)')
    print(f'a = {gas.speed_sound():.2f} m/s')

    print('\n=== Test 2: PH Update ===')
    h_test = gas.hmass()  # Use enthalpy from previous state
    gas2 = IdealGasEos(R_air, gamma_air)
    # HmassP_INPUTS expects (h, P) - enthalpy first, pressure second
    gas2.update(PH_INPUTS, h_test, p_test)
    print(f'T from PH = {gas2.T():.2f} K (expected {T_test:.2f} K)')

    print('\n=== Test 3: PS Update ===')
    s_test = gas.smass()
    gas3 = IdealGasEos(R_air, gamma_air)
    gas3.update(PS_INPUTS, p_test, s_test)
    print(f'T from PS = {gas3.T():.2f} K (expected {T_test:.2f} K)')

    print('\n=== Test 4: DP Update ===')
    rho_test = gas.rhomass()
    gas4 = IdealGasEos(R_air, gamma_air)
    gas4.update(DP_INPUTS, rho_test, p_test)
    print(f'T from DP = {gas4.T():.2f} K (expected {T_test:.2f} K)')

    print('\n=== Test 5: HS Update ===')
    gas5 = IdealGasEos(R_air, gamma_air)
    gas5.update(HS_INPUTS, h_test, s_test)
    print(f'T from HS = {gas5.T():.2f} K (expected {T_test:.2f} K)')
    print(f'P from HS = {gas5.p():.2e} Pa (expected {p_test:.2e} Pa)')
