from typing import Generic, Sequence, TypeVar

from adet.assembly import SystemAssembler
from adet.components import BaseComponent
from adet.components.connections import Inlet
from adet.equations.nondimensional import AbsoluteMachNumber, RelativeMachNumber
from adet.fluid.settings import FluidSettings

from adet.equations.fundamental import (
    MassAreaRelation,
    Kinematics,
    MeridionalUniform,
    TotalStaticMatching,
)

from adet.equations.definitions import CumMassFlow

from adet.equations.linkers import ComponentLinker, VariableAdder
from adet.tools.iter import grouper
from adet.tools.printing import print_header


T = TypeVar('T', bound=SystemAssembler)

# Equations that are to be defined at each single node
# of the network
_SINGLE_NODE_EQUATIONS = [
    # FOUNDATIONAL EQs - DO NOT REMOVE!
    MassAreaRelation,
    Kinematics,
    MeridionalUniform,
    TotalStaticMatching,
    # Courtesy definitions (the system is well posed w/o)
    CumMassFlow,
    AbsoluteMachNumber,
    RelativeMachNumber,
    VariableAdder,
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
        *components: BaseComponent,
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
        self._link_components(self.num_components)

    def _read_components(self, components: Sequence[BaseComponent]):
        for position, comp in enumerate(components):
            component_inlet_index = 2 * position
            component_outlet_index = 2 * position + 1

            self.system.add_boundary_conditions(
                comp.boundary_conditions,
                component_outlet_index,
            )

            for equation, node_pos in comp._equations.items():
                if isinstance(node_pos, int):
                    traslated_pos = component_inlet_index + node_pos
                else:
                    traslated_pos = [component_inlet_index + pos for pos in node_pos]

                self.system.add_equation(equation, traslated_pos)

    def _add_single_node_eqs(self, comp_stack_length: int):
        for node_idx in range(2 * comp_stack_length):
            # Single node relationships, make instances!
            [self.system.add_equation(eq(), node_idx) for eq in _SINGLE_NODE_EQUATIONS]

    def _link_components(self, comp_stack_length: int):
        # Nomenclature
        # ~~~~~~~~~~~~
        #        ___________________________________
        #          |         |          _________
        #          |         |  linker |         |
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

        link_node_couples = list(
            grouper(
                range(1, 2 * comp_stack_length),
                2,
                incomplete='ignore',
            ),
        )

        # Check that this is not empty (single row case)
        if link_node_couples:
            for nodes in link_node_couples:
                self.system.add_equation(ComponentLinker(), nodes)

    def print_structure(self):
        component_repr = ''
        for idx, comp in enumerate(self.components):
            comp_name = comp.name
            component_repr += (
                f' {{{{ (node {2 * idx})--|[ {comp_name} ]|'
                f'--(node {2 * idx + 1}) }}}} =='
            )

        print(component_repr)
