# === IMPORTS
from adet.fluid.ideal_eos import IdealGasState
import logging
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
from CoolProp import AbstractState
from numpy.typing import NDArray
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.fundamental import FreeVortexDistribution
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


class RowBackend:
    """Blade row network that is solved once and then re-solved with Newton.

    The geometry boundary conditions live in the known-parameters vector ``kn``.
    ``set_geometry`` overwrites entries of ``kn`` and ``solve`` runs the Newton
    rootfinder starting from the last converged solution.
    """

    def __init__(self):
        self._build()
        self._solve_initial()

    # === Setup
    def _build(self):
        # abs_state = AbstractState('HEOS', 'Air')
        abs_state = IdealGasState(1.4, 287, 2e-5)

        # *** Inlet conditions
        inlet = Inlet(
            boundary_conditions={
                # *** Inlet total conditions
                n0.oth.TotMassFlow: 10.0,
                n0.tot.Pressure: 10e5,
                n0.tot.Temperature: 500,
            }
        )

        casing = Shaft(100, is_constrained=True)
        shaft = Shaft(-1, is_constrained=False)

        self.stator = BladeRow(
            name='stator',
            shaft=casing,
            bound_cond={
                # *** Inlet geometry
                n0.geo.Rmid: 0.2,
                n0.geo.Height: 0.1,
                n0.geo.MeridionalAngle: Quantity(0, 'deg'),
                # *** Outlet geometry
                n1.geo.MeridionalAngle: Quantity(0, 'deg'),
                n1.geo.Height: 0.12,
                n1.geo.Rmid: 0.2,
                # *** Chord
                n1.geo.ChordAx: 0.1,
                n1.geo.NumBlades: 40,  # Number of blades
                n0.geo.MetalAngle: Quantity(30, 'deg'),
                n1.geo.MetalAngle: Quantity(-30, 'deg'),
            },
            extra_equations={
                ZeroDeviation(): 0,  # No incidence (design)
                TotalPressureLoss(0.9): (0, 1),  # Loss coefficient
                # ModifiedZweifel(): (0, 1),
            },
            spanwise_constants=[n1.geo.ChordAx],
        )

        # > Modify the rotor
        rotor = deepcopy(self.stator)  # Reuse the stator as template
        rotor.shaft = shaft  # Assign the rotating shaft
        rotor.name = 'rotor'

        self.stator.set_spanwise_constant(
            # Uniform inlet meridional velocity and streamtubes heights
            n0.kin.V_mer,
            n0.geo.HDistr,
        )

        fluid_settings = FluidSettings(
            fluid_state=abs_state,
            update_variables=(n0.stc.Pressure, n0.stc.Temperature),
        )

        self.ntw = ComponentNetwork(
            fluid_settings=fluid_settings,
            inlet=inlet,
            backend=CasadiSystem(num_span=1),
            components=[self.stator],
        )

        # Free vortex radial equilibrium
        if self.ntw.system.num_span > 1:
            self.stator.add_equation(FreeVortexDistribution(), 1)
            rotor.add_equation(FreeVortexDistribution(), 1)

        self.ntw.build()

        system = self.ntw.system
        self.x0 = system.get_guess(fallback=0.8)
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
