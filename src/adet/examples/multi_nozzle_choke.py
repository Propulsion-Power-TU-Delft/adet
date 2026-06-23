from typing import Literal
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.equations.base_equation import EquationBase
from adet.equations.fundamental import (
    ConstRelEnthalpy,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.equations.nondimensional import RelativeMachNumber
from adet.fluid.settings import FluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.losses.basic import IsentropicLink
from adet.solution import solve_optimization_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.tools.plotting import setup_mpl
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

system = CasadiSystem(1)


def add_node_equations(idx: int) -> dict:
    """Create equations for a single node."""
    return {
        Kinematics(): idx,
        MassAreaRelation(): idx,
        TotalStaticMatching(): idx,
        RelativeMachNumber(): idx,
        ConstRelEnthalpy(): (idx - 1, idx),
        MassConservation(): (idx - 1, idx),
        IsentropicLink(): (idx - 1, idx),
    }


def make_impulse_rotor(inlet_node: NodeVariables, outlet_node: NodeVariables):
    """Factory function to create an ImpulseRotor class for specific nodes."""

    class ImpulseRotor(EquationBase):
        """Rotor equation: constant static pressure across rotor."""

        def residual(
            self,
            p0: inlet_node.stc.Pressure.Hint,
            p1: outlet_node.stc.Pressure.Hint,
        ):
            return p0 - p1

    return ImpulseRotor()


def build_stator_rotor_stage(
    _stage_num: int,
    inlet_node_idx: int,
    equations_dict: dict,
    nodes_list: list,
    rotor_replacements: list,
) -> int:
    """
    Add a stator (nozzle) + rotor (impulse) stage to the system.

    Parameters
    ----------
    _stage_num : int
        Stage identifier (1, 2, ...)
    inlet_node_idx : int
        Node index of the stage inlet
    equations_dict : dict
        Dictionary to accumulate equations (modified in place)
    nodes_list : list
        List of NodeVariables (modified in place if new nodes needed)
    rotor_replacements : list
        List of (inlet_node, outlet_node, pos_tuple) for rotor equations

    Returns
    -------
    int
        Outlet node index of this stage
    """
    stator_outlet_idx = inlet_node_idx + 1
    rotor_outlet_idx = inlet_node_idx + 2

    # Extend nodes_list if necessary
    while len(nodes_list) <= rotor_outlet_idx:
        nodes_list.append(NodeVariables(len(nodes_list)))

    # Stator: nozzle equations
    equations_dict.update(add_node_equations(stator_outlet_idx))

    # Rotor: add node equations, then mark for replacement of enthalpy
    equations_dict.update(add_node_equations(rotor_outlet_idx))
    rotor_replacements.append(
        (
            nodes_list[stator_outlet_idx],
            nodes_list[rotor_outlet_idx],
            (stator_outlet_idx, rotor_outlet_idx),
        )
    )

    return rotor_outlet_idx


# Configure number of stages here
NUM_STAGES = 7

# Create inlet node
n0 = NodeVariables(0)
nodes = [n0]

# Build inlet equations
EQS = {
    Kinematics(): 0,
    MassAreaRelation(): 0,
    RelativeMachNumber(): 0,
    TotalStaticMatching(): 0,
}

# Build stages iteratively
rotor_replacements = []
current_outlet = 0
for stage_num in range(1, NUM_STAGES + 1):
    current_outlet = build_stator_rotor_stage(
        stage_num, current_outlet, EQS, nodes, rotor_replacements
    )

last_node_idx = current_outlet

BCS = {
    n0.tot.Pressure: 1e6,
    n0.tot.Temperature: 500,
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.kin.Omega: 0,
    n0.geo.RDistr: 0.1,
}


MULTIPLIER = 1.185
# MULTIPLIER = 1
MODE: Literal['lin', 'cum'] = 'lin'
IN_AREA = 0.1
FINAL_STAGE = 6
FINAL_AREA = 0.38


def generate_areas(last_idx):
    """Generate effective area for each node using multiplier mode."""
    area = IN_AREA
    for _ in range(last_idx + 1):
        yield area
        area *= MULTIPLIER


# Add area boundary conditions for all nodes
prev_area = IN_AREA
if MODE == 'cum':
    for i, (node, base_area) in enumerate(
        zip(nodes[: last_node_idx + 1], generate_areas(last_node_idx))
    ):
        effective_area = prev_area if i % 2 == 1 else base_area * 2
        prev_area = base_area
        BCS[node.geo.EffArea] = effective_area
elif MODE == 'lin':
    for i, node in enumerate(nodes[: last_node_idx + 1]):
        base_area = IN_AREA + (FINAL_AREA - IN_AREA) * i / 2 / FINAL_STAGE
        effective_area = prev_area if i % 2 == 1 else base_area * 2
        prev_area = base_area
        BCS[node.geo.EffArea] = effective_area


# NOTE: Stator + Rotor stages scheme
#
#   Stator     Rotor
#  ____        ____
#      \      |    |
#       \___  |____|
#  _ . _ . _ . _ . _ .
#

abs_state = DebugAbstractState('HEOS', 'Air')
idl_state = IdealGasState(1.4, 287, 2e-5)

fluid_model = FluidModel(abs_state)

system.fluid_settings = FluidSettings(
    fluid_model,
    (n0.stc.Pressure.Glob, n0.stc.Temperature.Glob),
)

[system.add_equation(eq, pos) for eq, pos in EQS.items()]

# Replace enthalpy equations with impulse rotor equations for each stage
for stator_node, rotor_node, position in rotor_replacements:
    system.remove_equation(ConstRelEnthalpy, position)
    system.add_equation(make_impulse_rotor(stator_node, rotor_node), position)

system.add_equalities(
    tuple(node.kin.Omega for node in nodes),
    tuple(node.geo.RDistr for node in nodes),
    tuple(node.kin.FlowAngleAbs for node in nodes),
)

system.add_boundary_conditions(BCS)

system.build()
input('Press enter to continue...')

# *** Optimizer formulation
obj_func = 1 / system.free_args_sym[nodes[1].oth.MassFlow]
# ***

x0 = system.get_scaled_guess(fallback=0.01)
kn = system.get_scaled_constraints()
bnd = system.get_arguments_bounds(
    {
        # Node limiters
        n0.stc.Pressure.Glob: (1, 1e7),
        n0.stc.Temperature.Glob: (110, 1e4),
        # Inlet Mach limit
        n0.kin.RelMach.Glob: (0.0, 1.0),
    },
    ignore_defaults=True,
)


solution, optimizer = solve_optimization_problem(
    system, obj_func, x0, kn, bnd, {'error_on_fail': False}
)

sol = solution['x'].toarray().flatten()  # type: ignore[attr-defined]
# rtfn = system.make_rootfinder('kinsol')
# solve_root_problem(rtfn, sol, kn)

data = system.sol_to_dict(sol)

pressure_ratio = data[nodes[last_node_idx].stc.Pressure] / data[n0.tot.Pressure]

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
setup_mpl(
    {
        'font.family': 'EB Garamond',
        'font.size': 20,
    }
)


# Sweep number of stages
SWEEP_STAGES = True
if SWEEP_STAGES:
    stage_nums = range(1, 8)  # Test 1 to 8 stages
    choke_pressure_ratios = []

    for num_stages in stage_nums:
        try:
            # Rebuild system with different number of stages
            system_sweep = CasadiSystem(1)

            # Build inlet equations
            EQS_sweep = {
                Kinematics(): 0,
                MassAreaRelation(): 0,
                RelativeMachNumber(): 0,
                TotalStaticMatching(): 0,
            }

            # Build stages
            rotor_replacements_sweep = []
            current_outlet = 0
            for stage_num in range(1, num_stages + 1):
                current_outlet = build_stator_rotor_stage(
                    stage_num,
                    current_outlet,
                    EQS_sweep,
                    nodes,
                    rotor_replacements_sweep,
                )

            last_node_idx_sweep = current_outlet

            # Boundary conditions
            BCS_sweep = {
                n0.tot.Pressure: 1e6,
                n0.tot.Temperature: 500,
                n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
                n0.kin.Omega: 0.0,
                n0.geo.RDistr: 0.1,
            }

            # Apply area boundary conditions
            prev_area_sweep = IN_AREA
            if MODE == 'cum':
                for i, (node, base_area) in enumerate(
                    zip(
                        nodes[: last_node_idx_sweep + 1],
                        generate_areas(last_node_idx_sweep),
                    )
                ):
                    effective_area = prev_area_sweep if i % 2 == 1 else base_area * 2
                    prev_area_sweep = base_area
                    BCS_sweep[node.geo.EffArea] = effective_area
            elif MODE == 'lin':
                for i, node in enumerate(nodes[: last_node_idx_sweep + 1]):
                    base_area = IN_AREA + (FINAL_AREA - IN_AREA) * i / 2 / FINAL_STAGE
                    effective_area = prev_area_sweep if i % 2 == 1 else base_area * 2
                    prev_area_sweep = base_area
                    BCS_sweep[node.geo.EffArea] = effective_area

            system_sweep.fluid_settings = FluidSettings(
                fluid_model,
                (n0.stc.Pressure.Glob, n0.stc.Temperature.Glob),
            )

            [system_sweep.add_equation(eq, pos) for eq, pos in EQS_sweep.items()]

            for stator_node, rotor_node, position in rotor_replacements_sweep:
                system_sweep.remove_equation(ConstRelEnthalpy, position)
                impulse_rotor = make_impulse_rotor(stator_node, rotor_node)
                system_sweep.add_equation(impulse_rotor, position)

            nodes_sweep = nodes[: last_node_idx_sweep + 1]
            system_sweep.add_equalities(
                tuple(node.kin.Omega for node in nodes_sweep),
                tuple(node.geo.RDistr for node in nodes_sweep),
                tuple(node.kin.FlowAngleAbs for node in nodes_sweep),
            )

            system_sweep.add_boundary_conditions(BCS_sweep)
            system_sweep.build()

            # Solve
            obj_func_sweep = 1 / system_sweep.free_args_sym[nodes[1].oth.MassFlow]
            x0_sweep = system_sweep.get_scaled_guess(fallback=0.01)
            kn_sweep = system_sweep.get_scaled_constraints()
            bnd_sweep = system_sweep.get_arguments_bounds(
                {
                    n0.stc.Pressure.Glob: (1, 1e7),
                    n0.stc.Temperature.Glob: (110, 1e4),
                    n0.kin.RelMach.Glob: (0.0, 1.4),
                },
                ignore_defaults=True,
            )

            opts = {'error_on_fail': True}
            sol_sweep, opt_sweep = solve_optimization_problem(
                system_sweep,
                obj_func_sweep,
                x0_sweep,
                kn_sweep,
                bnd_sweep,
                opts,
            )

            sol_array = sol_sweep['x'].toarray().flatten()  # type: ignore[attr-defined]
            data_sweep = system_sweep.sol_to_dict(sol_array)
            outlet_node = nodes[last_node_idx_sweep]
            p_outlet = float(data_sweep[outlet_node.stc.Pressure])
            p_in = float(data_sweep[n0.tot.Pressure])
            p_ratio = p_outlet / p_in
            choke_pressure_ratios.append(p_ratio)
            print(f'Stages: {num_stages}, Choke pressure ratio: {p_ratio:.4f}')
        except Exception as e:
            print(f'Stages: {num_stages}, Error: {e}')
            choke_pressure_ratios.append(None)

    # Plot results
    fig_stages, ax_stages = plt.subplots(figsize=(8, 6))
    valid_pairs = [
        (s, p) for s, p in zip(stage_nums, choke_pressure_ratios) if p is not None
    ]
    valid_stages = [s for s, p in valid_pairs]
    valid_ratios = [p for s, p in valid_pairs]

    ax_stages.plot(valid_stages, valid_ratios, 'o-', linewidth=2, markersize=8)
    ax_stages.set_xlabel('Number of Stator-Rotor Stages')
    ax_stages.set_ylabel(r'Choking Pressure Ratio ($p_e / p_{t,in}$)')
    ax_stages.grid(alpha=0.5)
    ax_stages.set_title('Choke Point Pressure Ratio vs. Number of Stages')
    plt.show()

    # Plot effective area distribution for the last stage configuration
    fig_area, ax_area = plt.subplots(figsize=(8, 6))
    node_indices = []
    area_values = []
    for i, node in enumerate(nodes):
        if i % 2 == 0:
            continue
        node_indices.append(i)
        area_values.append(BCS_sweep[node.geo.EffArea])

    ax_area.plot(
        node_indices,
        np.array(area_values),
        'o-',
        linewidth=2,
        markersize=8,
        color='steelblue',
    )
    ax_area.set_xlabel('Node Index')
    ax_area.set_ylabel(r'Effective Area (m$^2$)')
    ax_area.grid(alpha=0.5)
    ax_area.set_title(f'Effective Area Distribution ({num_stages} stages)')
    plt.close('all')
