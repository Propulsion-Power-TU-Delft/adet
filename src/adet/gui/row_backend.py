# === IMPORTS
import copy
import importlib
import logging
import math
import pickle
import re
import threading
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import CoolProp as cp
import numpy as np
import tomli_w
from numpy.typing import NDArray

from adet.assemblers import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.fundamental import FreeVortexDistribution
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, VarSpec

n0 = NodeVariables(0)
n1 = NodeVariables(1)

logger = logging.getLogger(__name__)
setup_logger(logger)


NODE_NAMES = ('inlet', 'outlet')  # TOML names of the nodes n0 and n1 of a row
CASING = 'casing'  # name of the non-rotating shaft
SOLVER_WALL_TIME = 2.0  # [s] limit for each of ipopt and kinsol in the initial solve


def _run_with_timeout(func: Callable[[], Any], timeout: float) -> Any:
    """Run ``func`` in a daemon thread; raise ``TimeoutError``
    if it outlasts ``timeout``.

    A hung thread cannot be killed, it is only abandoned.
    """
    result: dict[str, Any] = {}

    def target():
        try:
            result['value'] = func()
        except BaseException as err:
            result['error'] = err

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f'no result after {timeout} s')
    if 'error' in result:
        raise result['error']
    return result['value']


def _split_equation(path: str) -> tuple[str, str]:
    """``'losses.basic.ZeroDeviation'`` to the module ``adet.losses.basic`` and the
    class name ``ZeroDeviation``."""
    module, _, name = path.rpartition('.')
    if not module:
        raise ValueError(
            f'Equation {path!r} needs its module, like "losses.basic.{path}"'
        )
    return f'adet.{module}', name


def _eval_equations(equations: Sequence[Mapping[str, Any]]) -> dict[Any, Any]:
    """Extra equations of a row from TOML-like specs to ``{equation: nodes}``.

    Each spec has ``equation`` (class path inside ``adet``, like
    ``losses.basic.TotalPressureLoss``), ``position`` (an int or a sequence of
    nodes) and optionally ``parameters`` (positional list or keyword table).
    """
    built = {}
    for spec in equations:
        module, name = _split_equation(spec['equation'])
        try:
            cls = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            raise ValueError(f'Unknown equation {spec["equation"]!r}') from None
        params = spec.get('parameters', [])
        if isinstance(params, Mapping):
            equation = cls(**params)
        else:
            equation = cls(*params)
        position = spec['position']
        built[equation] = (
            tuple(position) if isinstance(position, Sequence) else position
        )
    return built


def _equation_code(spec: Mapping[str, Any]) -> str:
    """Source of the key of an extra equation, like ``TotalPressureLoss(0.9)``."""
    params = spec.get('parameters', [])
    if isinstance(params, Mapping):
        args = ', '.join(f'{k}={v!r}' for k, v in params.items())
    else:
        args = ', '.join(repr(v) for v in params)
    return f'{_split_equation(spec["equation"])[1]}({args})'


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
    # *** Axial length (the chord and meridional chord are results) and blades
    n1.geo.AxialLength: 0.1,
    n1.geo.NumBlades: 40,
    # *** Relative flow angles; the metal angles (camber lines) are results
    n0.kin.FlowAngleRel: math.radians(30),
    n1.kin.FlowAngleRel: math.radians(-30),
    # *** Rotational speed of the row
    n1.kin.Omega: 0.0,
}
# Extra equations of a new row: no incidence (the inlet metal angle follows the
# imposed relative flow angle), no deviation, and a total pressure loss
DEFAULT_EXTRA_EQUATIONS: list[dict[str, Any]] = [
    {'equation': 'losses.basic.ZeroDeviation', 'position': 0},
    {
        'equation': 'losses.basic.TotalPressureLoss',
        'position': [0, 1],
        'parameters': [0.9],
    },
]
# magnitude of the change in relative flow angle across an added row
FLOW_TURNING = math.radians(10)


@dataclass(frozen=True)
class FluidChoice:
    """Working fluid: an ideal gas, or a CoolProp fluid with its backend."""

    ideal: bool = True
    gamma: float = 1.4
    gas_constant: float = 287.0  # J/(kg K)
    viscosity: float = 2e-5  # Pa s
    name: str = 'Air'  # CoolProp fluid name
    backend: str = 'HEOS'  # CoolProp backend

    def make_state(self):
        """Fluid state object for the network."""
        if self.ideal:
            return IdealGasState(self.gamma, self.gas_constant, self.viscosity)
        from CoolProp import AbstractState

        return AbstractState(self.backend, self.name)


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
        fluid: FluidChoice | None = None,
        extra_equations: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    ):
        # Boundary conditions of each row, with the specs of the row's own nodes
        # (n0 = inlet, n1 = outlet) and values in base units. A row imposes either its
        # relative or its absolute flow angles, whichever specs it holds
        self.row_params = [dict(p) for p in row_params or [FIRST_ROW_PARAMS]]
        # Extra equations of each row as a list of {equation, position, parameters}
        # (the TOML form), padded with the defaults or cut to the number of rows
        extra = [[dict(s) for s in e] for e in extra_equations or []]
        extra += [
            copy.deepcopy(DEFAULT_EXTRA_EQUATIONS)
            for _ in range(len(self.row_params) - len(extra))
        ]
        self.extra_equations = extra[: len(self.row_params)]
        self.inlet_conditions = dict(inlet_conditions or INLET_CONDITIONS)
        # Values by spec (not by position) used as the initial guess where available
        self._guess: Mapping[VarSpec, NDArray] = guess or {}
        self.fluid = fluid or FluidChoice()
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
        abs_state = self.fluid.make_state()

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
                extra_equations=_eval_equations(self.extra_equations[index]),
                spanwise_constants=[n1.geo.AxialLength],
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
                'ipopt',
                {'error_on_fail': True, 'ipopt.max_wall_time': SOLVER_WALL_TIME},
            )
            sol = solve_root_problem(rtfn, self.x0, self.kn, suppress_output=True)
        except RuntimeError:
            # Bounded
            rtfn = system.make_rootfinder(
                'ipopt',
                {'error_on_fail': False, 'ipopt.max_wall_time': SOLVER_WALL_TIME},
            )
            sol = solve_root_problem(
                rtfn, self.x0, self.kn, self.bnd, suppress_output=False
            )

        # Kinsol, with Newton as a fallback if it fails or hangs
        # NOTE: CasADi's kinsol has no wall-time option, so it runs in a worker
        # thread that is abandoned after SOLVER_WALL_TIME
        rtfn = system.make_rootfinder('kinsol', {'max_iter': 100})
        try:
            sol = _run_with_timeout(
                lambda: solve_root_problem(rtfn, sol, self.kn), SOLVER_WALL_TIME
            )
        except (RuntimeError, TimeoutError) as err:
            logger.warning(f'KINSOL failed ({err}), trying Newton...')
            rtfn = system.make_rootfinder('newton', {'max_iter': 25})
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

    def rotor_coefficients(self, row: int) -> tuple[float, float, float]:
        """Work coefficient, flow coefficient and total-total efficiency of a row.

        Post-processing of the current solution, following ``WorkCoefficient``,
        ``FlowCoefficient`` and ``TotalTotalExpansionEfficiency`` (or the compression
        one if the row adds enthalpy). The speed is the blade speed at the outlet, so
        the row must turn.
        """
        node0, node1 = self.row_nodes(row)
        get = self.get_value
        h0_tot = get(node0.tot.Enthalpy)
        h1_tot = get(node1.tot.Enthalpy)
        u1 = get(node1.kin.BladeSpeed)
        work = (h1_tot - h0_tot) / u1**2
        flow = get(node0.kin.V_mer) / abs(u1)

        # Outlet enthalpy after an isentropic change to the outlet total pressure
        state = self.fluid.make_state()
        state.update(cp.PSmass_INPUTS, get(node1.tot.Pressure), get(node0.stc.Entropy))
        h_is1 = state.hmass()
        if h1_tot < h0_tot:
            eta = (h0_tot - h1_tot) / (h0_tot - h_is1)
        else:
            eta = (h_is1 - h0_tot) / (h1_tot - h0_tot)
        return work, flow, eta

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

    def solve_ipopt(self) -> bool:
        """Recover with Ipopt from the last solution. Returns False if it failed."""
        system = self.ntw.system
        try:
            try:
                # Unbounded
                rtfn = system.make_rootfinder(
                    'ipopt',
                    {'error_on_fail': True, 'ipopt.max_wall_time': SOLVER_WALL_TIME},
                )
                sol = solve_root_problem(rtfn, self._sol, self.kn, suppress_output=True)
            except RuntimeError:
                # Bounded
                rtfn = system.make_rootfinder(
                    'ipopt',
                    {'error_on_fail': True, 'ipopt.max_wall_time': SOLVER_WALL_TIME},
                )
                sol = solve_root_problem(
                    rtfn, self._sol, self.kn, self.bnd, suppress_output=True
                )
        except RuntimeError as err:
            logger.warning(f'Ipopt did not converge: {err}')
            return False

        if not np.all(np.isfinite(sol)):
            logger.warning('Ipopt returned a non-finite solution')
            return False

        self._sol = sol
        self.sol_dict = system.sol_to_dict(sol)
        return True

    # === Adding and removing rows
    def next_row_params(self, omega: float | None = None) -> dict[VarSpec, float]:
        """Boundary conditions of a row that would follow the last one.

        The inlet geometry comes from the link to the previous row. The row keeps
        the meridional angle, the height and the chord of the last row, and its
        endwalls stay parallel. It turns at ``omega`` (rad/s), or at the speed of the
        last row if none is given. It imposes relative flow angles, whatever
        the last row does; its outlet relative flow angle is turned by
        ``FLOW_TURNING`` from the relative flow angle leaving the last row, added or
        subtracted, whichever gives the smaller absolute angle.
        """
        _, last = self.row_nodes(self.num_rows - 1)
        get = self.get_value
        angle = get(last.geo.MeridionalAngle)
        height = get(last.geo.Height)
        # The row is a rectangle: the distance between the inlet and outlet centers
        # is half the height of the station lines
        chord = 0.5 * height
        angle_in = get(last.kin.FlowAngleRel)
        # Add or subtract the turning, whichever leaves the smaller absolute angle
        angle_out = min(
            (angle_in + FLOW_TURNING, angle_in - FLOW_TURNING),
            key=lambda value: abs(value),
        )
        return {
            n1.geo.MeridionalAngle: angle,
            n1.geo.Height: height,
            # The outlet center lies on the straight line through the previous outlet
            # center, normal to the station line, which is tilted by the meridional
            # angle (as drawn in the GUI). It is one chord away, so the radius drops
            # by chord * sin(angle) and the walls stay parallel: the row is a
            # rectangle in the meridional view, also for a radial row
            n1.geo.Rmid: get(last.geo.Rmid) - chord * math.sin(angle),
            # Axial part of that distance; MinimalMeridional gives the meridional chord
            n1.geo.AxialLength: chord * math.cos(angle),
            n1.geo.NumBlades: get(last.geo.NumBlades),
            n1.kin.FlowAngleRel: angle_out,
            n1.kin.Omega: get(last.kin.Omega) if omega is None else omega,
        }

    def add_row(self, params: Mapping[VarSpec, float]):
        """Append a row and solve the longer chain from scratch.

        ``params`` are the boundary conditions of the new row in its own nodes. The
        current geometry of the existing rows is kept. If the new network cannot be
        solved, the error is raised and this backend stays as it was.
        """
        self._rebuild([*self._current_row_params(), dict(params)])
        # Constructor pads the new row with no extra equations

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

    @staticmethod
    def _spec_code(spec: VarSpec) -> str:
        """Source code of a spec of node 0 or 1, like ``n1.geo.Rmid``."""
        node = (n0, n1)[spec.node]
        for container in ('kin', 'geo', 'ndim', 'oth', 'tot', 'stc', 'rlt'):
            instance = getattr(node, container)
            for name in dir(type(instance)):
                if isinstance(getattr(type(instance), name), VarSpec) and (
                    getattr(instance, name) == spec
                ):
                    return f'n{spec.node}.{container}.{name}'
        raise ValueError(f'No source name for {spec}')

    @staticmethod
    def _spec_from_code(key: str) -> VarSpec:
        """Inverse of ``_spec_code``: ``'n1.geo.Rmid'`` to its spec."""
        node, container, name = key.split('.')
        try:
            return getattr(getattr(NodeVariables(int(node[1:])), container), name)
        except AttributeError:
            raise ValueError(f'Unknown variable {key!r} in the TOML') from None

    @classmethod
    def _nest(cls, values: Mapping[VarSpec, float]) -> dict[str, Any]:
        """Values by spec as ``{node: {container: {name: value}}}``, for TOML tables."""
        nested: dict[str, Any] = {}
        for spec, value in values.items():
            node, container, name = cls._spec_code(spec).split('.')
            node = NODE_NAMES[int(node[1:])]
            nested.setdefault(node, {}).setdefault(container, {})[name] = value
        return nested

    @staticmethod
    def _node_index(name: str) -> int:
        if name not in NODE_NAMES:
            raise ValueError(f'Unknown node {name!r} in the TOML, use {NODE_NAMES}')
        return NODE_NAMES.index(name)

    @classmethod
    def _unnest(cls, nested: Mapping[str, Any]) -> dict[VarSpec, float]:
        """Inverse of ``_nest``."""
        return {
            cls._spec_from_code(f'n{cls._node_index(node)}.{container}.{name}'): value
            for node, containers in nested.items()
            for container, names in containers.items()
            for name, value in names.items()
        }

    def _shaft_groups(
        self, rows: Sequence[Mapping[VarSpec, float]]
    ) -> tuple[dict[str, float], list[str]]:
        """Shafts of the rows: speed by shaft name, and the shaft name of each row.

        Rows that do not turn sit on the casing, rows with the same speed share a
        shaft, named ``shaft0``, ``shaft1``... in order of appearance.
        """
        shafts = {CASING: 0.0}
        names = []
        for params in rows:
            omega = params[n1.kin.Omega]
            name = next((n for n, value in shafts.items() if value == omega), None)
            if name is None:
                name = f'shaft{len(shafts) - 1}'
                shafts[name] = omega
            names.append(name)
        return shafts, names

    def save_solution(self, path: Path):
        """Pickle the converged solution (values by spec), the guess of a rebuild."""
        with open(path, 'wb') as file:
            pickle.dump(self.sol_dict, file)

    def to_toml(self, solution_file: str | None = None) -> str:
        """The current setup as TOML: fluid, inlet conditions and the rows.

        It holds the same data as ``export_script``; ``from_toml`` rebuilds it. With
        ``solution_file``, a ``[solution]`` section names the pickled solution (see
        ``save_solution``) that ``from_toml`` starts from, relative to the TOML file.
        """
        rows = self._current_row_params()
        shafts, row_shafts = self._shaft_groups(rows)
        setup = {
            'fluid': asdict(self.fluid),
            # The inlet conditions are all at the inlet node, so it is left implicit
            'inlet': self._nest(
                {spec: self.get_value(spec) for spec in self.inlet_conditions}
            )[NODE_NAMES[0]],
            # Written as [shaft.casing], [shaft.shaft0]...; the rows refer to them
            'shaft': {
                name: {'Omega': omega, 'fixed': True} for name, omega in shafts.items()
            },
            # Written as [[row]] with sub-tables like [row.outlet.kin]. The speed of
            # the row is that of its shaft
            'row': [
                {
                    'shaft': shaft,
                    # Written as [[row.extra_equations]] with equation, position
                    # and optionally parameters
                    'extra_equations': extra,
                    **self._nest(
                        {s: v for s, v in params.items() if s != n1.kin.Omega}
                    ),
                }
                for shaft, params, extra in zip(row_shafts, rows, self.extra_equations)
            ],
        }
        if solution_file is not None:
            setup['solution'] = {'file': solution_file}
        text = tomli_w.dumps(setup)
        # tomli_w writes every array one item per line with a trailing comma, which
        # TOML formatters keep split; short lists like [0, 1] are put back inline
        return re.sub(
            r'\[\n((?:[ \t]+[^\[\]\n]+,\n)+)\]',
            lambda m: (
                '['
                + ', '.join(v.strip().rstrip(',') for v in m[1].split('\n')[:-1])
                + ']'
            ),
            text,
        )

    @classmethod
    def from_toml(cls, text: str, base_dir: Path = Path('.')) -> 'RowBackend':
        """Build and solve the chain described by TOML written by ``to_toml``.

        The pickled solution of the ``[solution]`` section, if any, is the initial
        guess; its file is relative to ``base_dir``.
        """
        setup = tomllib.loads(text)
        guess = None
        if 'solution' in setup:
            solution = base_dir / setup['solution']['file']
            try:
                with open(solution, 'rb') as file:
                    guess = pickle.load(file)
            except FileNotFoundError:
                # Without the pickle the chain is solved directly
                logger.warning(f'Solution {solution} not found, solving directly')
        shafts = setup['shaft']
        row_params = []
        extra_equations = []
        for index, row in enumerate(setup['row']):
            row = dict(row)
            # A row without the key has no extra equations (not the defaults)
            extra_equations.append(row.pop('extra_equations', []))
            name = row.pop('shaft', CASING)
            if name not in shafts:
                raise ValueError(f'Row {index} is on unknown shaft {name!r}')
            if not shafts[name].get('fixed', True):
                raise ValueError(f'Shaft {name!r}: only fixed speeds are supported')
            params = cls._unnest(row)
            params[n1.kin.Omega] = shafts[name]['Omega']
            row_params.append(params)
        return cls(
            row_params=row_params,
            inlet_conditions=cls._unnest({NODE_NAMES[0]: setup['inlet']}),
            guess=guess,
            fluid=FluidChoice(**setup['fluid']),
            extra_equations=extra_equations,
        )

    def export_script(self, solution_file: str | None = None) -> str:
        """Python script that builds and solves the chain as it is now.

        The geometry, flow angles, speeds, inlet conditions and fluid are the current
        ones. It follows ``_build`` and ``_solve_initial``. With ``solution_file``
        (see ``save_solution``) it starts from that pickled solution, which it looks
        for next to the script, and otherwise from a generic guess.
        """

        def code(spec: VarSpec) -> str:
            node, rest = self._spec_code(spec).split('.', 1)
            return f'{("inlet", "outlet")[spec.node]}.{rest}'

        fluid = self.fluid
        if fluid.ideal:
            state = (
                f'IdealGasState({fluid.gamma!r}, {fluid.gas_constant!r}, '
                f'{fluid.viscosity!r})'
            )
        else:
            state = f'AbstractState({fluid.backend!r}, {fluid.name!r})'

        def dict_lines(values: Mapping[VarSpec, float], indent: str) -> str:
            return ''.join(
                f'{indent}{code(spec)}: {value!r},\n' for spec, value in values.items()
            )

        inlet = {spec: self.get_value(spec) for spec in self.inlet_conditions}
        current = self._current_row_params()
        shafts, row_shafts = self._shaft_groups(current)
        rows = [
            {s: v for s, v in params.items() if s != n1.kin.Omega} for params in current
        ]
        # One import per module, with the classes of all the rows
        by_module: dict[str, set[str]] = {}
        for extra in self.extra_equations:
            for spec in extra:
                module, name = _split_equation(spec['equation'])
                by_module.setdefault(module, set()).add(name)
        equation_imports = [
            f'from {module} import {", ".join(sorted(names))}'
            for module, names in sorted(by_module.items())
        ]
        guess_lines = (
            [
                '# Converged solution of the GUI, values by variable',
                f"with open(Path(__file__).parent / {solution_file!r}, 'rb') as file:",
                '    guess = pickle.load(file)',
            ]
            if solution_file is not None
            else ['guess = {}']
        )
        lines = [
            '"""Blade row chain exported from the ADeT GUI."""',
            '',
            'import logging',
            'import pickle',
            'from pathlib import Path',
            '',
            'import CoolProp as cp  # noqa: F401',
            'from CoolProp import AbstractState  # noqa: F401',
            '',
            'from adet.assemblers import CasadiSystem',
            'from adet.components import BladeRow, Inlet',
            'from adet.components.connections import Shaft',
            'from adet.components.network import ComponentNetwork',
            'from adet.equations import EquationBase',
            'from adet.fluid.ideal_eos import IdealGasState  # noqa: F401',
            'from adet.fluid.settings import FluidSettings',
            *equation_imports,
            'from adet.solution import solve_root_problem',
            'from adet.tools.loggers import setup_logger',
            'from adet.variables import NodeVariables',
            '',
            'logger = logging.getLogger(__name__)',
            'setup_logger(logger)',
            '',
            'inlet = NodeVariables(0)',
            'outlet = NodeVariables(1)',
            '',
            '# Inlet total conditions',
            'INLET_CONDITIONS = {',
            dict_lines(inlet, '    ').rstrip('\n'),
            '}',
            '',
            '# Shafts: rotational speed [rad/s] and whether it is imposed',
            'SHAFTS = {',
            *(f'    {name!r}: ({omega!r}, True),' for name, omega in shafts.items()),
            '}',
            '',
            '# Shaft of each row',
            f'ROW_SHAFTS = {row_shafts!r}',
            '',
            '# Boundary conditions of each row in its own nodes (inlet inlet, outlet outlet)',
            'ROW_PARAMS = [',
        ]
        for params in rows:
            lines += ['    {', dict_lines(params, '        ').rstrip('\n'), '    },']
        lines += [
            ']',
            '',
            '# Extra equations of each row, {Equation(): nodes}',
            'ROW_EXTRA_EQUATIONS: list[dict[EquationBase, int | tuple[int, ...]]] = [',
            *(
                '    {'
                + ', '.join(
                    f'{_equation_code(spec)}: '
                    f'{tuple(spec["position"]) if isinstance(spec["position"], Sequence) else spec["position"]!r}'
                    for spec in extra
                )
                + '},'
                for extra in self.extra_equations
            ),
            ']',
            '',
            f'state = {state}',
            'inlet_comp = Inlet(boundary_conditions=INLET_CONDITIONS)',
            '',
            'shafts = {',
            '    name: Shaft(omega, is_constrained=fixed)',
            '    for name, (omega, fixed) in SHAFTS.items()',
            '}',
            'rows = []',
            'for index, params in enumerate(ROW_PARAMS):',
            '    row = BladeRow(',
            "        name=f'row{index}',",
            '        shaft=shafts[ROW_SHAFTS[index]],',
            '        bound_cond=params,',
            '        extra_equations=ROW_EXTRA_EQUATIONS[index],',
            '        spanwise_constants=[outlet.geo.AxialLength],',
            '    )',
            '    if index == 0:',
            '        row.set_spanwise_constant(inlet.kin.V_mer, inlet.geo.HDistr)',
            '    rows.append(row)',
            '',
            'ntw = ComponentNetwork(',
            '    fluid_settings=FluidSettings(',
            '        fluid_state=state,',
            '        update_variables=(inlet.stc.Pressure, inlet.stc.Temperature),',
            '    ),',
            '    inlet=inlet_comp,',
            '    backend=CasadiSystem(num_span=1),',
            '    components=rows,',
            ')',
            'ntw.build()',
            '',
            'system = ntw.system',
            *guess_lines,
            'x0 = system.get_guess(guess, fallback=0.8)',
            'kn = system.get_boundary_conds()',
            'rootfinder = system.make_rootfinder(',
            "    'ipopt', {'error_on_fail': True, 'ipopt.max_wall_time': 5.0}",
            ')',
            'sol = solve_root_problem(rootfinder, x0, kn, suppress_output=True)',
            "rootfinder = system.make_rootfinder('newton', {'max_iter': 25})",
            'sol = solve_root_problem(rootfinder, sol, kn, suppress_output=True)',
            'sol_dict = system.sol_to_dict(sol)',
            '',
            'if __name__ == "__main__":',
            '    for index in range(len(rows)):',
            '        node_in, node_out = NodeVariables(2 * index), NodeVariables(2 * index + 1)',
            '        print(',
            "            f'row {index}: Vm = {sol_dict[node_out.kin.V_mer][0]:.2f} m/s, '",
            "            f'Pt out = {sol_dict[node_out.tot.Pressure][0]:.1f} Pa'",
            '        )',
            '',
        ]
        return '\n'.join(lines)

    def set_fluid(self, fluid: FluidChoice):
        """Change the working fluid and solve the chain again from the current one.

        If the new network cannot be solved, the error is raised and this backend
        stays as it was.
        """
        self._rebuild(self._current_row_params(), fluid)

    def _rebuild(
        self, rows: list[dict[VarSpec, float]], fluid: FluidChoice | None = None
    ):
        """Replace the chain by ``rows``, keeping the inlet and the current solution."""
        inlet_conditions = {
            spec: self.get_value(spec) for spec in self.inlet_conditions
        }
        # The positions of the knowns and of the free variables change with the nodes,
        # so the converged solution goes in by spec as the initial guess
        new = RowBackend(
            rows,
            inlet_conditions,
            guess=self.sol_dict,
            fluid=fluid or self.fluid,
            extra_equations=self.extra_equations,
        )
        # Only reached when the new network solved, so nothing is half updated
        self.__dict__.update(new.__dict__)
