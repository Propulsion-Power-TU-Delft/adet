from typing import Generic, Sequence, TypeVar

from adet.assembly import SystemAssembler
from adet.components import BaseComponent
from adet.components.connections import Inlet
from adet.fluid.settings import FluidSettings

from adet.equations.geometrical import MeridionalUniform
from adet.equations.fundamental import MassAreaRelation, Kinematics, TotalStaticMatching
from adet.equations.nondimensional import AbsoluteMachNumber, RelativeMachNumber
from adet.equations.definitions import CumMassFlow
from adet.equations.special import ThermoVarsAdder

from adet.tools.printing import print_header


T = TypeVar('T', bound=SystemAssembler)

# Equations that are to be defined at each single node
# of the network
_SINGLE_NODE_EQUATIONS = [
    # *** FOUNDATIONAL EQs - DO NOT REMOVE!
    MassAreaRelation,
    Kinematics,
    MeridionalUniform,
    TotalStaticMatching,
    # *** Courtesy definitions
    # -> the system is well posed w/o them
    # if they are not mentioned in other equations
    CumMassFlow,
    AbsoluteMachNumber,
    RelativeMachNumber,
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
        """
        Network of turbomachinery components
        """
        print_header()

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
                self.system.add_invariants(
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
        #          |         |          _________
        #          |         |  invar. |         |
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
                    self.system.add_invariants(
                        (
                            f'{var + str(outlet_node_idx)}',
                            f'{var + str(inlet_node_idx)}',
                        )
                    )

    def print_structure(self):
        component_repr = '@ = node\n\nInlet == @0'

        for idx, comp in enumerate(self.components):
            comp_name = comp.name
            component_repr += f'\n== @{2 * idx}--|[ {comp_name} ]|--@{2 * idx + 1} =='

        print(component_repr)
