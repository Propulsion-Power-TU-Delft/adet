from collections import defaultdict
from typing import Generic, Mapping, Sequence, TypeVar

from adet.assembly import SystemAssembler
from adet.components import BaseComponent
from adet.components.blade_row import BladeRow
from adet.constants import ArrayLike
from adet.fluid.settings import FluidSettings
from adet.components.connections import Inlet, Shaft

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
        self._link_shafts()

    def _read_components(self, components: Sequence[BaseComponent]):
        for comp_index, comp in enumerate(components):
            inlet_node_idx = 2 * comp_index
            outlet_node_idx = 2 * comp_index + 1

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
        #        N |         | N     N |         | N
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
            # NOTE: Starts from 1 because the inlet is not linked
            for comp_index, component in enumerate(self.components[1:], 1):
                inlet_node_idx = 2 * comp_index  # Of the current component
                outlet_node_idx = inlet_node_idx - 1  # Of the previous component

                # Link from previous node
                for var in component._from_previous_node:
                    self.system.add_equalities(
                        (
                            f'{var + str(outlet_node_idx)}',
                            f'{var + str(inlet_node_idx)}',
                        )
                    )

    def _link_shafts(self):
        """Link all of the outlet omegas that belong to the same shaft"""
        #     ._____.    ._____.    ._____.    ._____.
        #     |     |    |     |    |     |    |     |
        #     |  S  |    |  R  |    |  S  |    |  R  |
        #  ___|_____|____|_____|____|_____|____|_____|___
        #     N-----N    N-----N    N-----N    N-----N <- Linked by component
        #       =0             |      =0             |
        #                      +---------------------+ <- Linked by this method

        shafts_outnodes: dict[Shaft, list[int]] = defaultdict(list)
        for comp_index, component in enumerate(self.components):
            inlet_node_idx = 2 * comp_index  # Of the current component
            outlet_node_idx = inlet_node_idx + 1  # Of the previous component

            if isinstance(component, BladeRow):
                # Constrained ones are enfored as omega constraints
                if not component._shaft.is_constrained:
                    shafts_outnodes[component._shaft].append(outlet_node_idx)

        for nodes in shafts_outnodes.values():
            linked_omegas = tuple(f'kin_omega{n}' for n in nodes)
            if len(linked_omegas) > 1:
                self.system.add_equalities(linked_omegas)

    def build(self, scaled: bool = True):
        self.system.build(scaled)

        # Write nodes on components for easier access
        for comp_idx, comp in enumerate(self.components):
            inlet_node_idx = 2 * comp_idx  # Of the current component
            outlet_node_idx = inlet_node_idx + 1  # Of the previous component

            comp.inlet_node = self.system.nodes[inlet_node_idx]
            comp.outlet_node = self.system.nodes[outlet_node_idx]

    def get_scaled_guess(self, manual_values: Mapping[str, ArrayLike] = {}):
        """Simple passthrough"""
        return self.system.get_scaled_guess(manual_values)

    def get_scaled_constraints(self):
        """Simple passthrough"""
        return self.system.get_scaled_constraints()

    def get_arguments_bounds(self, custom_bounds: dict[str, tuple[float, float]] = {}):
        physical_bounds = {}
        for comp_idx, comp in enumerate(self.components):
            if not isinstance(comp, BladeRow):
                continue
            else:
                node_in_idx = 2 * comp_idx
                node_out_idx = node_in_idx + 1

                if comp.row_type == 'stator':
                    physical_bounds[f'kin_U{node_in_idx}'] = (-0.1, 0.1)
                    physical_bounds[f'kin_U{node_out_idx}'] = (-0.1, 0.1)

        custom_bounds = {**physical_bounds, **custom_bounds}
        return self.system.get_arguments_bounds(custom_bounds)

    def print_structure(self):
        component_repr = '@ = node\n\nInlet == @0'

        for comp_idx, comp in enumerate(self.components):
            comp_name = comp.name
            component_repr += (
                f'\n== @{2 * comp_idx}--|[ {comp_name} ]|--@{2 * comp_idx + 1} =='
            )

        print(component_repr)
