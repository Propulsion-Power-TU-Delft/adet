from typing import Generic, Sequence, TypeVar
from art import tprint

from adet.assembly import SystemAssembler
from adet.components import BaseComponent
from adet.components.connections import Inlet
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings

from adet.equations.fundamental import (
    MassAreaRelation,
    Kinematics,
    MeridionalUniform,
    TotalStaticMatching,
    CumMassFlow,
)

from adet.equations.linkers import ComponentLinker
from adet.tools.iter import grouper
from adet.tools.printing import print_header


T = TypeVar('T', bound=SystemAssembler)


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
        self.system.settings = fluid_settings

        self.num_components = len(components)
        self.components = components

        self.system = backend
        self.system.settings = fluid_settings

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
        self.system.add_equation(CumMassFlow(), 0)

        for node in range(2 * comp_stack_length):
            # Single node relationships
            self.system.add_equation(MassAreaRelation(), node)
            self.system.add_equation(Kinematics(), node)
            self.system.add_equation(MeridionalUniform(), node)
            self.system.add_equation(TotalStaticMatching(), node)

            if isinstance(self.system.settings.model, AnalyticalFluidModel):
                eos_equations = self.system.settings.model.get_equations()

                for eq in eos_equations:
                    self.system.add_equation(eq, node)

    def _link_components(self, comp_stack_length: int):
        # Nomenclature
        # ~~~~~~~~~~~~
        #        _________________________________
        #          |         |        _________
        #          |         |       |         |
        #   V      |         |       |         |
        #  -->   0 |  ROW 0  | 1   2 |  ROW 1  | 3
        #          |         |       |         |
        #        __|_________|_______|_________|__
        #        /////////////////////////////////
        #        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        #        /////////////////////////////////
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

        # Check if this is not empty (single row case)
        if link_node_couples:
            for nodes in link_node_couples:
                self.system.add_equation(ComponentLinker(), nodes)

    def build_network(self, scale_equations: bool = True):
        self.system.build(scale_equations)
