from typing import Generic, Mapping, Sequence, TypeVar

from adet.assembly import SystemAssembler
from adet.components import BaseComponent
from adet.components.blade_row import BladeRow
from adet.constants import ArrayLike
from adet.fluid.settings import FluidSettings
from adet.components.connections import Inlet

from adet.equations.special import ThermoVarsAdder
from adet.equations.geometrical import AnnulusAreas
from adet.equations.nondimensional import AbsoluteMachNumber, RelativeMachNumber
from adet.equations.fundamental import (
    Kinematics,
    TotalMassFlow,
    MassAreaRelation,
    TotalArea,
    TotalStaticMatching,
)

from adet.tools.printing import print_logo

T = TypeVar('T', bound=SystemAssembler)

# Equations that are to be defined at each single node
# of the network
_SINGLE_NODE_EQUATIONS = [
    # *** FOUNDATIONAL EQs - DO NOT REMOVE!
    Kinematics,
    AnnulusAreas,
    MassAreaRelation,
    TotalStaticMatching,
    # *** Courtesy definitions
    # -> the system is well posed w/o them
    # if they are not mentioned in other equations
    TotalArea,
    TotalMassFlow,
    AbsoluteMachNumber,
    RelativeMachNumber,
    # *** Spezial
    ThermoVarsAdder,
]


# NOTE: I use composition and not inheritance
# because I want the network to use both the
# Casadi and Jax backends
class ComponentNetwork(Generic[T]):
    def __init__(
        self,
        fluid_settings: FluidSettings,
        inlet: Inlet,
        backend: T,
        components: Sequence[BaseComponent],
    ) -> None:
        """Network of turbomachinery components"""
        print_logo()

        self.system = backend
        self.system.fluid_settings = fluid_settings

        self.components = components
        self.num_components = len(components)

        self.system = backend
        self.system.fluid_settings = fluid_settings

        # Add inlet boundary conditions
        self.system.add_boundary_conditions(inlet.boundary_conditions, 0)

        # Read components
        self._read_components(components)

        self._add_single_node_eqs(self.num_components)
        self._link_components()

    def _read_components(self, components: Sequence[BaseComponent]):
        for position, comp in enumerate(components):
            inlet_node_idx = 2 * position
            outlet_node_idx = 2 * position + 1

            for var in comp._constant_variables:
                self.system.add_equalities(
                    (f'{var + str(inlet_node_idx)}', f'{var + str(outlet_node_idx)}')
                )

            self.system.add_boundary_conditions(
                comp.in_constraints,
                inlet_node_idx,
            )

            self.system.add_boundary_conditions(
                comp.out_constraints,
                outlet_node_idx,
            )

            for equation, node_pos in comp._equations.items():
                if isinstance(node_pos, int):
                    traslated_pos = inlet_node_idx + node_pos
                else:
                    traslated_pos = [inlet_node_idx + pos for pos in node_pos]

                self.system.add_equation(equation, traslated_pos)

    def _add_single_node_eqs(self, comp_stack_length: int):
        for node_idx in range(2 * comp_stack_length):
            # Single node relationships, make instances!
            [self.system.add_equation(eq(), node_idx) for eq in _SINGLE_NODE_EQUATIONS]

    def _link_components(self):
        # Nomenclature
        # ~~~~~~~~~~~~
        #        ___________________________________
        #          |         |         ,_________,
        #          |         |  eqlts. |         |
        #          |         |    ^    |         |
        #   V      |  ROW 0  |    |    |  ROW 1  |
        #  -->   0 |         | 1 === 2 |         | 3   <- NODES
        #        + |         | +     + |         | +
        #        __|_________|_________|_________|__
        #        ///////////////////////////////////
        #        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        #        ///////////////////////////////////
        #       _ . _ . _ . _ . _ . _ . _ . _ . _ . _
        #
        # * components couples : (0, 1), (2, 3), ...
        # * link couples : (1, 2), ...

        # Add invariants from one node to the next
        if len(self.components) > 1:
            for comp_index, component in enumerate(self.components[1:], 1):
                inlet_node_idx = 2 * comp_index  # Of the current component
                outlet_node_idx = inlet_node_idx - 1  # Of the previous component
                for var in component._from_previous_node:
                    self.system.add_equalities(
                        (
                            f'{var + str(outlet_node_idx)}',
                            f'{var + str(inlet_node_idx)}',
                        )
                    )

    def build(self, scaled: bool = True):
        self.system.build(scaled)

        # Write nodes on components for easier access
        for comp_index, component in enumerate(self.components):
            inlet_node_idx = 2 * comp_index  # Of the current component
            outlet_node_idx = inlet_node_idx + 1  # Of the previous component

            component.inlet_node = self.system.nodes[inlet_node_idx]
            component.outlet_node = self.system.nodes[outlet_node_idx]

    def get_scaled_guess(self, manual_values: Mapping[str, ArrayLike] = {}):
        """Simple passthrough"""
        physical_guesses = {}
        for cmp_idx, comp in enumerate(self.components):
            if not isinstance(comp, BladeRow):
                continue
            else:
                node_in_idx = 2 * cmp_idx
                node_out_idx = node_in_idx + 1

                if comp.row_type == 'stator':
                    physical_guesses[f'kin_U{node_in_idx}'] = 0.0
                    physical_guesses[f'kin_U{node_out_idx}'] = 0.0
                    physical_guesses[f'kin_alpha{node_in_idx}'] = 0.0
                    physical_guesses[f'kin_alpha{node_out_idx}'] = 1.20
                elif comp.row_type == 'rotor':
                    physical_guesses[f'kin_alpha{node_in_idx}'] = 1.20
                    physical_guesses[f'kin_alpha{node_in_idx}'] = 0.0

        manual_values = {**physical_guesses, **manual_values}
        return self.system.get_scaled_guess(manual_values)

    def get_scaled_constraints(self):
        """Simple passthrough"""
        return self.system.get_scaled_constraints()

    def get_arguments_bounds(self, custom_bounds: dict[str, tuple[float, float]] = {}):
        physical_bounds = {}
        for cmp_idx, comp in enumerate(self.components):
            if not isinstance(comp, BladeRow):
                continue
            else:
                node_in_idx = 2 * cmp_idx
                node_out_idx = node_in_idx + 1

                if comp.row_type == 'stator':
                    physical_bounds[f'kin_U{node_in_idx}'] = (-20, 20)
                    physical_bounds[f'kin_U{node_out_idx}'] = (-20, 20)

        custom_bounds = {**physical_bounds, **custom_bounds}
        return self.system.get_arguments_bounds(custom_bounds)

    def print_structure(self):
        component_repr = '@ = node\n\nInlet == @0'

        for idx, comp in enumerate(self.components):
            comp_name = comp.name
            component_repr += f'\n== @{2 * idx}--|[ {comp_name} ]|--@{2 * idx + 1} =='

        print(component_repr)
