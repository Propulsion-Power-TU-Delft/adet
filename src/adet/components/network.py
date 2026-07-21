from collections import defaultdict
from typing import Any, Generic, Literal, Sequence, TypeVar

from numpy.typing import NDArray

from adet.assemblers import SystemAssembler, SystemSharedData
from adet.components import BaseComponent
from adet.components.blade_row import BladeRow
from adet.components.connections import Inlet, Shaft
from adet.constants import AdetArray
from adet.equations.base_equation import EquationBase
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalArea,
    TotalMassFlow,
    TotalStaticMatching,
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.nondimensional import AbsoluteMachNumber, RelativeMachNumber
from adet.fluid.settings import FluidSettings
from adet.tools.iter import ensure_tuple
from adet.variables import KinematicVariables
from adet.varspec import DEF_NODE, VarSpec

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

        self.system = backend
        self.system.fluid_settings = fluid_settings

        self.inlet = inlet
        self.components = components
        self.num_components = len(components)

        self.system = backend
        self.system.fluid_settings = fluid_settings

        self._frozen_equations: dict[EquationBase, tuple[int, ...]] = {}

        self._add_single_node_eqs(self.num_components)

        # First write step, for manipulation before building
        self._write_to_system(inlet, components)

    def _get_abs_indices(self, component: BaseComponent):
        comp_idx = self.components.index(component)
        inlet_node_idx = 2 * comp_idx
        outlet_node_idx = 2 * comp_idx + 1
        return inlet_node_idx, outlet_node_idx

    def _write_to_system(
        self,
        inlet: Inlet,
        components: Sequence[BaseComponent],
    ):
        # Add inlet boundary conditions
        self.system.add_boundary_conditions(inlet.boundary_conditions)
        # Read components
        self._dispatch_components(components)
        self._link_components()
        self._link_shafts()
        self._frozen_equations = self.system.data.equations.copy()

    def _dispatch_components(self, components: Sequence[BaseComponent]):
        for comp in components:
            comp.attach_system(self.system)
            inl_idx, out_idx = self._get_abs_indices(comp)

            # Add equations
            for equation, node_pos in comp._equations.items():
                node_pos = ensure_tuple(node_pos)
                traslated_pos = [inl_idx + pos for pos in node_pos]
                self.system.add_equation(equation, traslated_pos)

            # Map boundary conditions to their absolute nodes
            mapped_bcond = {}
            for spec, val in comp._boundary_conditions.items():
                abs_node = comp.system_maps[self.system][spec.node]
                mapped_bcond[spec.at_node(abs_node)] = val
            self.system.add_boundary_conditions(mapped_bcond)

            # Write equalities (constant variables)
            for spec in comp._const_variables:
                equality = (
                    spec.at_node(inl_idx),
                    spec.at_node(out_idx),
                )
                self.system.add_equalities(equality)

            for spec in comp._spanwise_constants:
                abs_idx = inl_idx if spec.node == 0 else out_idx
                abs_spec = spec.at_node(abs_idx)
                self.system.add_spanwise_constants(abs_spec)

    def _add_single_node_eqs(self, comp_stack_length: int):
        for node_idx in range(2 * comp_stack_length):
            # Single node relationships, make instances!
            for eq in _SINGLE_NODE_EQUATIONS:
                self.system.add_equation(eq(), node_idx)

    def _link_components(self):
        # Nomenclature
        # ~~~~~~~~~~~~
        #          ._________.         ._________.
        #          |         |  eqlts. |         |
        #          |         |    ^    |         |
        #   V      |  ROW 0  |    |    |  ROW 1  |
        #  -->   0 |         | 1 === 2 |         | 3   <- NODES
        #        __|_________|_________|_________|__
        #        ///////////////////////////////////
        #        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        #       _ . _ . _ . _ . _ . _ . _ . _ . _ . _

        # Add invariants from one node to the next
        if len(self.components) > 1:
            # NOTE: 'prev' starts from 1, 'next' ends at -1
            self._link_node_variables('prev')
            self._link_node_variables('next')

    def _link_node_variables(self, mode: Literal['prev', 'next']):
        comps = self.components[1:] if mode == 'prev' else self.components[:-1]
        for comp in comps:
            inl_idx, out_idx = self._get_abs_indices(comp)
            if mode == 'prev':
                variables, left_idx, right_idx = (
                    comp._from_prev_node,
                    inl_idx - 1,
                    inl_idx,
                )
            else:
                variables, left_idx, right_idx = (
                    comp._from_next_node,
                    out_idx,
                    out_idx + 1,
                )
            for spec in variables:
                self.system.add_equalities(
                    (spec.at_node(left_idx), spec.at_node(right_idx))
                )

    def _link_shafts(self):
        """Link all of the outlet omegas that belong to the same shaft"""
        #     ._____.    ._____.    ._____.    ._____.
        #     |     |    |     |    |     |    |     |
        #     |  S  |    |  R  |    |  S  |    |  R  |
        #  ___|_____|____|_____|____|_____|____|_____|___
        #     N-----N    N-----N    N-----N    N-----N <- Linked by blade row
        #       =0             |      =0             |
        #                      +---------------------+ <- Linked by this method

        shafts_outnodes: dict[Shaft, list[int]] = defaultdict(list)
        for comp in self.components:
            _, out_idx = self._get_abs_indices(comp)

            if isinstance(comp, BladeRow):
                if comp.shaft is None:
                    raise AttributeError('Missing shaft')
                # Constrained ones are enfored as omega constraints: they
                # are already enforced for each row that belongs to that shaft
                if not comp.shaft.is_constrained:
                    shafts_outnodes[comp.shaft].append(out_idx)

        for nodes in shafts_outnodes.values():
            omega_genspec = KinematicVariables(DEF_NODE).Omega
            linked_omegas = tuple(omega_genspec.at_node(n) for n in nodes)
            if len(linked_omegas) > 1:
                self.system.add_equalities(linked_omegas)

    def build(self, scaled: bool = True):
        self.system.build(scaled)

    def get_guess(
        self, manual_values: dict[VarSpec, AdetArray] | None = None
    ) -> list[NDArray]:
        """Simple passthrough"""
        return self.system.get_guess(manual_values or {})

    def get_boundary_cond(self) -> list[NDArray]:
        """Simple passthrough"""
        return self.system.get_boundary_conds()

    @property
    def data(self) -> SystemSharedData:
        """Convenience access method"""
        return self.system.data

    def get_bounds(
        self, custom_bounds: dict[VarSpec, tuple[float, float]] | None = None
    ) -> tuple[Any, Any]:
        return self.system.get_bounds(custom_bounds or {})

    def print_structure(self):
        component_repr = '@ = node\n\nInlet == @0'

        for comp in self.components:
            inl_idx, out_idx = self._get_abs_indices(comp)
            comp_name = comp.name
            component_repr += f'\n== @{inl_idx}--|[ {comp_name} ]|--@{out_idx} =='

        print(component_repr)
