"""
This file contains the logic for handling variables across the adet library.
In general we will call 'constraints' all the quantities that are fixed by design
parameter, while 'variables' are all the quantities that are not fixed.
"""

from typing import ClassVar, Optional, Literal, get_args, Iterator
from itertools import combinations
import logging

import numpy as np
from numpy.typing import NDArray

from pint.facets.plain import PlainQuantity
from pint.registry import Quantity

from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.registries import DefaultUnitsRegistry
from adet.constants import ArrayLike
from adet.tools.plotting import plot_velocity_triangles


logger = logging.getLogger(__name__)


class VariableContainer:
    # Define valid types for that container, if any
    valid_types: ClassVar[tuple[str, ...] | None] = None
    _def_units = DefaultUnitsRegistry()

    def __init__(self, spanwise_stations: int = 1):
        self._variables: dict[str, PlainQuantity] = {}
        self._constraints: dict[str, PlainQuantity] = {}

        if spanwise_stations % 2 == 0:
            # Round to the closest odd number
            self._spanwise_stations = spanwise_stations | 1
            logger.warning(
                f'Rounding up the spanwise_stations {spanwise_stations}'
                f'to the nearest odd number ({self._spanwise_stations})'
            )
        else:
            self._spanwise_stations = spanwise_stations

    @property
    def constraints(self) -> dict[str, PlainQuantity]:
        """Get all constrained variables stored in the container."""
        return self._constraints

    @property
    def variables(self) -> dict[str, PlainQuantity]:
        """Get all free variables stored in the container."""
        return self._variables

    @property
    def all_quantities(self):
        """Return all variables in the container"""
        return {**self._variables, **self._constraints}

    def _validate_var_type(self, var_type: str):
        # If valid types are given, check them
        if self.valid_types:
            self._check_type_validity(var_type)

        if var_type in self.all_quantities:
            raise AttributeError(
                f'Variable {var_type} already exists as a constrained variable '
                f'in the container, skipping...'
            )

    def _validate_magnitude(self, magnitude: ArrayLike) -> NDArray:
        """
        Validate that a magnitude array has the correct shape for the container.

        Single values are expanded to match the number of spanwise stations.
        Array values must have a length matching the number of spanwise stations.

        Parameters
        ----------
        magnitude : ArrayLike
            The magnitude to validate.

        Returns
        -------
        NDArray
            The validated magnitude as a NumPy array with proper dimensions.

        Raises
        ------
        ValueError
            If the length of the magnitude array doesn't match the spanwise stations.
        """

        DTYPE = np.float64

        new_value = np.atleast_1d(magnitude)
        new_length = len(np.array(new_value))

        if new_length != self._spanwise_stations:
            if new_length == 1:
                logger.debug('Found single span magnitude, expanding to correct length')
                mag_validated = np.ones(self._spanwise_stations) * new_value[0]
            else:
                raise ValueError(
                    f'Variable length mismatch: spanwise stations '
                    f'{self._spanwise_stations} != variable length {new_length}'
                )
        else:
            mag_validated = np.asarray(magnitude, DTYPE)

        return mag_validated

    def _check_type_validity(self, var_type: str):
        if not self.valid_types:
            return

        if var_type in self.valid_types:
            return
        else:
            valid_types_str = '\n'.join(self.valid_types)
            raise AttributeError(
                f'Unknown variable type `{var_type}`, accepted entries '
                f'are:\n{valid_types_str}'
            )

    def _add_variable_helper(
        self,
        var_type: str,
        magnitude: Optional[ArrayLike],
        units: str | None,
        is_fixed: bool,
    ) -> None:
        if units is None:
            units = self._def_units[var_type]

        if magnitude is None:
            magnitude = np.full(self._spanwise_stations, np.nan)

        self._validate_var_type(var_type)
        mag_validated = self._validate_magnitude(magnitude)

        qty = Quantity(mag_validated, units)

        if is_fixed:
            self._constraints[var_type] = qty
        else:
            self._variables[var_type] = qty

    def add_variable(
        self,
        var_type: str,
        magnitude: Optional[float | ArrayLike] = None,
        units: str | None = None,
    ) -> None:
        self._add_variable_helper(var_type, magnitude, units, is_fixed=False)

    def add_constraint(
        self, var_type: str, magnitude: float | ArrayLike, units: str | None = None
    ) -> None:
        self._add_variable_helper(var_type, magnitude, units, is_fixed=True)

    def set_value(self, var_type: str, magnitude: ArrayLike) -> None:
        """This assumes base units"""
        if var_type not in self.all_quantities:
            self.add_variable(var_type)

        var = self.get(var_type)
        var.ito_base_units()
        mag_verified = self._validate_magnitude(magnitude)
        var._magnitude = mag_verified

    def change_status(self, var_type: str, fixed: bool):
        """
        Change the status of a variable, fixed or free

        Parameters
        ----------
        var_type: str
            Type of the variable you want to change, e.g. `'p'`
        fixed: bool
            True if you want to make the variable fixed, False if free
        """
        if fixed:
            if var_type in self._constraints:
                pass
            else:
                self._constraints[var_type] = self._variables.pop(var_type)
        else:
            if var_type in self._variables:
                pass
            else:
                self._variables[var_type] = self._constraints.pop(var_type)

    def remove_quantity(self, var_type):
        self._constraints.pop(var_type, None)
        self._variables.pop(var_type, None)

    def get(self, var_type: str) -> PlainQuantity:
        return self.all_quantities[var_type]

    def __getattr__(self, var_type: str) -> NDArray:
        return self.get(var_type).to_base_units().magnitude

    def __contains__(self, key):
        """
        Check if a variable type exists in the container.

        Parameters
        ----------
        key : str
            The variable type to check.

        Returns
        -------
        bool
            True if the variable exists, False otherwise.

        Examples
        --------
        >>> vc = VariableContainer()
        >>> vc.add_variable('p')
        >>> 'p' in vc
        True
        >>> 'T' in vc
        False
        """
        return key in self._variables

    def __iter__(self):
        """
        Iterate over variable types and their Quantity objects.

        Returns
        -------
        Iterable[tuple[str, Quantity]]
            Iterator over (var_type, Quantity) pairs.

        Examples
        --------
        >>> vc = VariableContainer()
        >>> vc.add_variable('p')
        >>> vc.add_variable('T')
        >>> for var_type, var in vc:
        ...     print(var_type)
        p
        T
        """
        return iter(self._variables.items())

    def __str__(self) -> str:
        """
        Generate string representation of the container with all variables.

        Returns
        -------
        str
            A multiline string with each variable on a separate line.

        Examples
        --------
        >>> vc = VariableContainer()
        >>> vc.add_variable('p', 101325, 'Pa')
        >>> print(vc)  # Shows formatted variable information
        """
        return ('\n\n').join(
            [f'{name} |> {var:.2f}' for name, var in self.all_quantities.items()]
        )


KinematicVariable = Literal[
    'V',
    'Vm',
    'Vt',
    'W',
    'Wt',
    'Wm',
    'U',
    'beta',
    'alpha',
    'omega',
]


class KinematicContainer(VariableContainer):
    valid_types = get_args(KinematicVariable)

    def plot(self, geo):
        return plot_velocity_triangles(self, geo)


ThermoVariable = Literal[
    'p',
    'T',
    'smass',
    'hmass',
    'umass',
    'rhomass',
    'speed_sound',
    'Q',
]


class ThermostateContainer(VariableContainer):
    valid_types = get_args(ThermoVariable)

    def __init__(self, spanwise_stations: int, fluid_settings: FluidSettings):
        super().__init__(spanwise_stations)
        self._sett = fluid_settings

    def _build_priorities(self) -> dict[str, int]:
        priorities = {}
        MAX_PRIO = 100
        MIN_PRIO = 10
        updt_vars = self._sett.update_variables

        for qty in self.all_quantities:
            if qty not in updt_vars:
                priorities[qty] = -1
            elif qty in self._constraints:
                priorities[qty] = MAX_PRIO - updt_vars.index(qty)
            elif qty in self._variables:
                priorities[qty] = MIN_PRIO - updt_vars.index(qty)

        return priorities

    def _identify_valid_pairs(self) -> Iterator[tuple[str, ...]] | None:
        updt_len = self._sett.update_length
        updt_vars = self._sett.update_variables

        # if isinstance(self._sett, AnalyticalFluidModel):
        #     return None

        valid_variables = set(self.all_quantities).intersection(updt_vars)

        if len(valid_variables) < updt_len:
            raise RuntimeError(
                f'Insufficient variables to define a valid pair, minimum is {updt_len}'
            )

        return combinations(valid_variables, updt_len)

    def choose_pair(self) -> tuple[str, ...] | None:
        """These are not sorted"""
        valid_pairs = self._identify_valid_pairs()

        if valid_pairs is None:
            return None

        priorities = self._build_priorities()

        pair_priorities = {
            pair: sum(priorities[p] for p in pair) for pair in valid_pairs
        }

        best_pair = max(
            pair_priorities,
            key=lambda x: pair_priorities.get(x, -1),
        )

        return best_pair


if __name__ == '__main__':
    import CoolProp as cp
    from adet.fluid.settings import FluidModel

    n_span = 11

    var1 = Quantity(n_span * [10.0], units='m/s')
    var2 = Quantity(np.ones(n_span) * 13, units='bar')

    var1.ito('ft/min')
    var1.to_base_units()

    magnitude_a = 1 + 1 / 2 * np.random.ranf(n_span)
    magnitude_b = 300 + 50 * np.random.ranf(n_span)

    vc = VariableContainer(spanwise_stations=n_span)

    vc.add_constraint('p', magnitude_a, 'bar')
    vc.add_variable('T', magnitude_b)
    vc.add_variable('hmass')

    # Update in place, this assumes base units!
    vc.set_value('p', 66666)

    # ThermoState
    sett = FluidSettings(
        ExternalFluidModel(
            cp.AbstractState(
                'HEOS',
                'R134a',
            ),
        ),
        ('hmass', 'p', 'T'),
        2,
    )

    tst = ThermostateContainer(n_span, sett)

    tst.add_constraint('p', magnitude_a, 'bar')
    tst.add_variable('T', magnitude_b)
    tst.add_variable('hmass')

    print(f'Chosen pair is {tst.choose_pair()}')
