"""
FlowNode module for adet

This module provides the FlowNode class which encapsulates the complete thermo-kinematic
state of a fluid at a specific location. It manages three thermodynamic states
(static, total, and relative total) along with fluid kinematics, enabling conversion
between these states and solving for the complete flow state given partial information.

The FlowNode is a key building block for simulating and analyzing fluid flow in
turbomachinery components.
"""

import logging
from typing import Literal, Sequence, get_args, Optional

# Ext libraries
from pint import Quantity

# Internal imports
from adet.tools.strings import get_arg_state, get_arg_type
from adet.constants import NodeStatesNames, ArrayLike
from adet.variables import VariableContainer, KinematicContainer, ThermostateContainer

from adet.fluid.settings import FluidSettings
from adet.fluid.properties import GasPropertiesMixin

logger = logging.getLogger(__name__)


class FlowNode(GasPropertiesMixin):
    instance_counter = 0

    def __init__(
        self,
        settings: FluidSettings,
        spanwise_stations: int = 1,
        node_name: Optional[str] = None,
    ):
        """
        Initialize a FlowNode with the specified gas settings.

        Parameters
        ----------
        settings : RealGasSettings or IdealGasSettings
            Configuration parameters for the gas model, including
            real or ideal gas properties and number of spanwise stations.
        node_name: str | None = None
            Provide a custom identifier to the node, if none is provided,
            the instance (int) count of the node instance is used, converted
            to a string
        """
        self.__class__.instance_counter += 1

        if node_name:
            self.identifier = node_name
        else:
            # Use the instance number of the node as identifier
            self.identifier = str(self.__class__.instance_counter)

        self.kin = KinematicContainer(spanwise_stations)

        self.stc = ThermostateContainer(spanwise_stations, settings)
        self.tot = ThermostateContainer(spanwise_stations, settings)
        self.rlt = ThermostateContainer(spanwise_stations, settings)

        self.geo = VariableContainer(spanwise_stations)
        self.oth = VariableContainer(spanwise_stations)

    def write_to_node(self, args_to_write: dict[str, ArrayLike], fixed: bool) -> None:
        for arg, value in args_to_write.items():
            state_id = get_arg_state(arg)
            state_obj = self.fetch_state(state_id)
            var_type = get_arg_type(arg)

            state_obj.set_value(var_type, value)
            state_obj.change_status(var_type, fixed)

    def _get_variables_helper(
        self,
        states: tuple[NodeStatesNames, ...],
        status: Literal['variables', 'constraints', 'all_quantities'],
    ) -> dict[str, Quantity]:
        """
        Helper method to gather variables from the different variable containers
        """
        if not states:
            state_names = get_args(NodeStatesNames)

        state_objects = {name: self.fetch_state(name) for name in state_names}

        variables = {}

        # Do not add double value
        qties_ids = []

        for state_name, state_obj in state_objects.items():
            REGISTRY = {
                'variables': state_obj.variables,
                'constraints': state_obj.constraints,
                'all_quantities': state_obj.all_quantities,
            }

            source = REGISTRY[status]
            qties_ids += [id(v) for v in variables.values()]
            variables.update(
                {f'{state_name}_{var}': val for var, val in source.items()}
            )

        return variables

    def get_variables(self, *state_names: NodeStatesNames):
        return self._get_variables_helper(state_names, status='variables')

    def get_constraints(self, *state_names: NodeStatesNames):
        return self._get_variables_helper(state_names, status='constraints')

    def get_all_quantities(self, *state_names: NodeStatesNames):
        return self._get_variables_helper(state_names, status='all_quantities')

    def fetch_state(self, state_id: NodeStatesNames) -> VariableContainer:
        if state_id not in get_args(NodeStatesNames):
            raise AttributeError(
                f'Unknown state {state_id}, valid states '
                f'are {get_args(NodeStatesNames)}'
            )
        return getattr(self, state_id)

    def get_update_variables(self) -> dict[str, tuple[str, ...]]:
        """
        Return the update variables, UNSORTED
        """
        STATES = {
            'stc': self.stc,
            'tot': self.tot,
            'rlt': self.rlt,
        }

        update_vars = {}

        for st_name, st_obj in STATES.items():
            # Get the valid pair
            pair = st_obj.choose_pair()

            if not pair:
                raise ValueError('No pair found')

            update_vars[st_name] = pair

        return update_vars

    def read_from_node(
        self, var_specs: Sequence[str], create_missing: bool = False
    ) -> list[Quantity]:
        output_variables = []
        for arg in var_specs:
            var_state = get_arg_state(arg)
            var_type = get_arg_type(arg)
            state_obj = self.fetch_state(var_state)

            # If create missing is explicitly set, create the variable
            if create_missing and var_type not in state_obj.all_quantities:
                state_obj.add_variable(var_type)

            variable = state_obj.get(var_type)
            output_variables.append(variable)

        return output_variables

    def create_vars(self, *variables_to_add: str) -> None:
        """
        Simpler syntax for just adding a series of variables, identified
        with, <state>_<var_type> as in `read_from_node`
        """
        self.read_from_node(variables_to_add, create_missing=True)

    def __str__(self):
        """
        Generate a string representation of the FlowNode.
        Returns
        -------
        str
            A formatted string showing all states and variables
        """
        return f"""FlowNode `{self.identifier}`: (ID {id(self)})

╭──────────────╮
│ STATIC STATE │
╰──────────────╯
{self.stc}

╭─────────────╮
│ TOTAL STATE │
╰─────────────╯
{self.tot}

╭──────────────────╮
│ REL. TOTAL STATE │
╰──────────────────╯
{self.rlt}

╭────────────╮
│ KINEMATICS │
╰────────────╯
{self.kin}

╭──────────╮
│ GEOMETRY │
╰──────────╯
{self.geo}

╭────────────╮
│ OTHER VARS │
╰────────────╯
{self.oth}"""
