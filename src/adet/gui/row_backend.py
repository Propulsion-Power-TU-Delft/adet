# === IMPORTS
import logging
import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from adet.assemblers import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.fundamental import FreeVortexDistribution
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.losses.basic import (
    TotalPressureLoss,
    ZeroDeviation,
)
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, VarSpec

n0 = NodeVariables(0)
n1 = NodeVariables(1)

logger = logging.getLogger(__name__)
setup_logger(logger)


INLET_CONDITIONS: dict[VarSpec, float] = {
    n0.oth.TotMassFlow: 50.0,
    n0.tot.Pressure: 10e5,
    n0.tot.Temperature: 500,
}
FIRST_ROW_PARAMS: dict[VarSpec, float] = {
    # *** Inlet geometry
    n0.geo.Rmid: 0.2,
    n0.geo.Height: 0.1,
    n0.geo.MeridionalAngle: 0.0,
    # *** Outlet geometry
    n1.geo.MeridionalAngle: 0.0,
    n1.geo.Height: 0.12,
    n1.geo.Rmid: 0.2,
    # *** Chord and blades
    n1.geo.ChordAx: 0.1,
    n1.geo.NumBlades: 40,
    # *** Relative flow angles; the metal angles (camber lines) are results
    n0.kin.FlowAngleRel: math.radians(30),
    n1.kin.FlowAngleRel: math.radians(-30),
    # *** Rotational speed of the row
    n1.kin.Omega: 0.0,
}
FLOW_TURNING = math.radians(
    10
)  # outlet minus inlet relative flow angle of an added row


class RowBackend:
    """Chain of blade rows that is solved once and then re-solved with Newton.

    Row ``k`` goes from node ``2k`` to node ``2k + 1``. The inlet geometry of every
    row but the first is linked to the outlet of the previous row, so ``row_params``
    only holds it for the first row. The geometry boundary conditions live in the
    known-parameters vector ``kn``. ``set_geometry`` overwrites entries of ``kn``
    and ``solve`` runs the Newton rootfinder starting from the last converged
    solution. ``add_row`` appends a row and rebuilds the network.
    """

    def __init__(
        self,
        row_params: Sequence[Mapping[VarSpec, float]] | None = None,
        inlet_conditions: Mapping[VarSpec, float] | None = None,
        guess: Mapping[VarSpec, NDArray] | None = None,
    ):
        # Boundary conditions of each row, with the specs of the row's own nodes
        # (n0 = inlet, n1 = outlet) and values in base units. A row imposes either its
        # relative or its absolute flow angles, whichever specs it holds
        self.row_params = [dict(p) for p in row_params or [FIRST_ROW_PARAMS]]
        self.inlet_conditions = dict(inlet_conditions or INLET_CONDITIONS)
        # Values by spec (not by position) used as the initial guess where available
        self._guess: Mapping[VarSpec, NDArray] = guess or {}
        self._build()
        self._solve_initial()

    @property
    def num_rows(self) -> int:
        return len(self.row_params)

    @staticmethod
    def row_nodes(row: int) -> tuple[NodeVariables, NodeVariables]:
        """Inlet and outlet node variables of a row."""
        return NodeVariables(2 * row), NodeVariables(2 * row + 1)

    def is_absolute(self, row: int, node: int) -> bool:
        """Whether the angle at a node (0 = inlet, 1 = outlet) of the row is absolute.

        The inlet of a following row takes the kind of the previous outlet, as it is
        the same station.
        """
        if node == 0 and row > 0:
            return self.is_absolute(row - 1, 1)
        return (n0, n1)[node].kin.FlowAngleAbs in self.row_params[row]

    def angle_spec(self, row: int, node: int) -> VarSpec:
        """Flow angle (of the kind imposed there) at a node (0 or 1) of the row."""
        kin = self.row_nodes(row)[node].kin
        return kin.FlowAngleAbs if self.is_absolute(row, node) else kin.FlowAngleRel

    @staticmethod
    def _with_angle_kind(
        params: Mapping[VarSpec, float], node: int, absolute: bool
    ) -> dict[VarSpec, float]:
        """Copy of ``params`` with the flow angle at ``node`` of the given kind."""
        kin = (n0, n1)[node].kin
        old, new = (
            (kin.FlowAngleRel, kin.FlowAngleAbs)
            if absolute
            else (kin.FlowAngleAbs, kin.FlowAngleRel)
        )
        return {(new if spec == old else spec): value for spec, value in params.items()}

    # === Setup
    def _build(self):
        # abs_state = AbstractState('HEOS', 'Air')
        abs_state = IdealGasState(1.4, 287, 2e-5)

        inlet = Inlet(boundary_conditions=dict(self.inlet_conditions))

        self.rows: list[BladeRow] = []
        for index, params in enumerate(self.row_params):
            bound_cond = {
                spec: value for spec, value in params.items() if spec != n1.kin.Omega
            }
            row = BladeRow(
                name=f'row{index}',
                shaft=Shaft(params[n1.kin.Omega], is_constrained=True),
                bound_cond=bound_cond,
                extra_equations={
                    # No incidence: the inlet metal angle follows the flow, which the
                    # imposed relative flow angles set. No deviation either (base)
                    ZeroDeviation(): 0,
                    TotalPressureLoss(0.9): (0, 1),  # Loss coefficient
                    # ModifiedZweifel(): (0, 1),
                },
                spanwise_constants=[n1.geo.ChordAx],
            )
            if index == 0:
                row.set_spanwise_constant(
                    # Uniform inlet meridional velocity and streamtubes heights
                    n0.kin.V_mer,
                    n0.geo.HDistr,
                )
            self.rows.append(row)

        fluid_settings = FluidSettings(
            fluid_state=abs_state,
            update_variables=(n0.stc.Pressure, n0.stc.Temperature),
        )

        self.ntw = ComponentNetwork(
            fluid_settings=fluid_settings,
            inlet=inlet,
            backend=CasadiSystem(num_span=1),
            components=self.rows,
        )

        # Free vortex radial equilibrium
        if self.ntw.system.num_span > 1:
            for row in self.rows:
                row.add_equation(FreeVortexDistribution(), 1)

        self.ntw.build()

        system = self.ntw.system
        self.x0 = system.get_guess(self._guess, fallback=0.8)
        self.kn = system.get_boundary_conds()
        self.bnd = system.get_bounds(
            {
                n0.geo.Chord.Glob: (0.0, 1e5),
                n0.kin.V_mag.Glob: (0.0, 500.0),
                n0.stc.Pressure.Glob: (10.0, 13e5),
                n0.stc.Temperature.Glob: (60.0, 500),
            },
            ignore_defaults=False,
        )

        # Position of every boundary condition inside ``kn``
        self._kn_index = {spec: i for i, spec in enumerate(system.data.boun_cond)}

    def _solve_initial(self):
        system = self.ntw.system

        # Ipopt
        try:
            # Unbounded
            rtfn = system.make_rootfinder(
                'ipopt', {'error_on_fail': True, 'ipopt.max_wall_time': 5}
            )
            sol = solve_root_problem(rtfn, self.x0, self.kn, suppress_output=True)
        except RuntimeError:
            # Bounded
            rtfn = system.make_rootfinder('ipopt', {'error_on_fail': False})
            sol = solve_root_problem(
                rtfn, self.x0, self.kn, self.bnd, suppress_output=False
            )

        # Kinsol
        rtfn = system.make_rootfinder('kinsol')
        sol = solve_root_problem(rtfn, sol, self.kn)

        self._sol: NDArray = sol
        self.sol_dict = system.sol_to_dict(sol)

        # Newton, used for every following update
        # NOTE: CasADi's newton has no wall-time option, so the work is capped by
        # the iteration count instead
        self._newton = system.make_rootfinder('newton', {'max_iter': 25})

    # === Interaction
    def get_value(self, spec: VarSpec) -> float:
        """Scalar value of any variable in the current solution (base units)."""
        if spec in self._kn_index:
            # sol_dict holds the boundary conditions as they were at build time
            idx = self._kn_index[spec]
            return float(self.kn[idx][0] * self.ntw.system.constraints_scaling[idx])
        return float(self.sol_dict[spec][0])

    def set_geometry(self, values: Mapping[VarSpec, float]):
        """Overwrite boundary conditions in ``kn`` (base units: m, rad)."""
        scales = self.ntw.system.constraints_scaling
        for spec, value in values.items():
            idx = self._kn_index[spec]
            self.kn[idx] = (
                np.atleast_1d(value) / scales[idx] * np.ones_like(self.kn[idx])
            )

    def solve(self) -> bool:
        """Run Newton from the last solution. Returns False if it did not converge."""
        try:
            sol = solve_root_problem(
                self._newton, self._sol, self.kn, suppress_output=True
            )
        except RuntimeError as err:
            logger.warning(f'Newton did not converge: {err}')
            return False

        if not np.all(np.isfinite(sol)):
            logger.warning('Newton returned a non-finite solution')
            return False

        self._sol = sol
        self.sol_dict = self.ntw.system.sol_to_dict(sol)
        return True

    # === Adding and removing rows
    def next_row_params(self, omega: float | None = None) -> dict[VarSpec, float]:
        """Boundary conditions of a row that would follow the last one.

        The inlet geometry comes from the link to the previous row. The row keeps
        the meridional angle, the height and the axial chord of the last row, and its
        endwalls stay parallel. It turns at ``omega`` (rad/s), or at the speed of the
        last row if none is given. It imposes relative flow angles, whatever
        the last row does; its outlet relative flow angle is turned
        by ``FLOW_TURNING`` from the relative flow angle leaving the last row.
        """
        _, last = self.row_nodes(self.num_rows - 1)
        get = self.get_value
        angle = get(last.geo.MeridionalAngle)
        chord = get(last.geo.ChordAx)
        angle_in = get(last.kin.FlowAngleRel)
        return {
            n1.geo.MeridionalAngle: angle,
            n1.geo.Height: get(last.geo.Height),
            # The station lines lean downstream for a positive angle (as drawn in the
            # GUI), so the mean line moves inwards; the walls stay parallel
            n1.geo.Rmid: get(last.geo.Rmid) - chord * math.tan(angle),
            n1.geo.ChordAx: chord,
            n1.geo.NumBlades: get(last.geo.NumBlades),
            n1.kin.FlowAngleRel: angle_in + FLOW_TURNING,
            n1.kin.Omega: get(last.kin.Omega) if omega is None else omega,
        }

    def add_row(self, params: Mapping[VarSpec, float]):
        """Append a row and solve the longer chain from scratch.

        ``params`` are the boundary conditions of the new row in its own nodes. The
        current geometry of the existing rows is kept. If the new network cannot be
        solved, the error is raised and this backend stays as it was.
        """
        self._rebuild([*self._current_row_params(), dict(params)])

    def remove_last_row(self):
        """Drop the last row and solve the shorter chain from scratch.

        The current geometry of the remaining rows is kept. If the network cannot be
        solved, the error is raised and this backend stays as it was.
        """
        if self.num_rows < 2:
            raise ValueError('The first row cannot be removed')
        self._rebuild(self._current_row_params()[:-1])

    def set_angle_mode(self, row: int, node: int, absolute: bool):
        """Make a row impose the absolute or relative flow angle at a node instead.

        The angles of the current solution become the new boundary conditions and the
        system is rebuilt. If the new network cannot be solved, the error is raised and
        this backend stays as it was. The inlet of a following row imposes nothing.
        """
        if absolute == self.is_absolute(row, node):
            return
        rows = self._current_row_params()
        rows[row] = {
            spec: self.get_value(spec.at_node(2 * row + spec.node))
            for spec in self._with_angle_kind(self.row_params[row], node, absolute)
        }
        self._rebuild(rows)

    def _current_row_params(self) -> list[dict[VarSpec, float]]:
        """Boundary conditions of every row as they are now."""
        return [
            {
                spec: self.get_value(spec.at_node(2 * index + spec.node))
                for spec in row_params
            }
            for index, row_params in enumerate(self.row_params)
        ]

    def _rebuild(self, rows: list[dict[VarSpec, float]]):
        """Replace the chain by ``rows``, keeping the inlet and the current solution."""
        inlet_conditions = {
            spec: self.get_value(spec) for spec in self.inlet_conditions
        }
        # The positions of the knowns and of the free variables change with the nodes,
        # so the converged solution goes in by spec as the initial guess
        new = RowBackend(rows, inlet_conditions, guess=self.sol_dict)
        # Only reached when the new network solved, so nothing is half updated
        self.__dict__.update(new.__dict__)
