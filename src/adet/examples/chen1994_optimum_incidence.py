"""
Rotor-inlet incidence for radial-inflow turbines (Chen & Baines, 1994),
solved as an isolated single-node model.

    H. Chen and N. C. Baines, "The aerodynamic loading of radial and
    mixed-flow turbines," Int. J. Mech. Sci., vol. 36, no. 1, pp. 63-79,
    1994. (``docs/chen1994 The aerodynamic loading of radial and
    mixed-flow turbines.pdf``)

The paper's central result is that the commonly assumed "zero incidence"
rotor-inlet design (relative flow angle = blade metal angle) is wrong:
just as at a centrifugal compressor impeller exit, the flow *slips*
relative to the blades. This is exactly the mechanism
``adet.equations.control_volumes.OptimalIncidenceRadialInflowTurbine``
implements as a ``DeviationModel`` -- see its docstring for the full
derivation and for the two slip-factor correlations it supports
(``'chen'`` = Eq. 8, the paper's own improved correlation; ``'stanitz'``
= Eq. 4, the older centrifugal-compressor correlation it is compared
against).

Importantly, the slip factor mu itself is a purely geometric/design
correlation (function of blade number Z, cone angle and blade angle
only), but the resulting *incidence* is not: it depends on the actual
flow arriving at the rotor (meridional velocity, blade speed), since
slip subtracts a fixed velocity deficit ``(1 - mu) * U`` from the
ideal swirl rather than pinning the flow angle to a single "design"
value. This script demonstrates both:

1. Sweeping the blade number Z at the theoretical Chen & Baines
   optimum-loading operating point (Eqs. 2, 5, 7a-7b, valid for the
   common zero blade angle case) reproduces the paper's Figs 3 and 6
   (inlet loading coefficient and optimum incidence vs. blade number)
   for both correlations.
2. Sweeping the flow coefficient Cm2/U2 at *fixed* Z shows that the
   incidence this equation predicts changes substantially with the
   operating point -- it is not a fixed function of geometry alone.

This is a convenient standalone sanity check before wiring the equation
into a full rotor model (see ``radial_inflow_turbine_blade_loading.py``
and ``t100_radial_inflow_turbine.py``, where the rotor's actual,
operating-point-dependent incidence is what results).
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from adet.assemblers import CasadiSystem
from adet.equations.control_volumes import OptimalIncidenceRadialInflowTurbine
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

n0 = NodeVariables(0)

# Purely radial rotor inlet (cone angle = 90 deg, the "90-degree IFR
# turbine" convention) with radial (zero-angle) blades -- the case the
# paper's Figs 3 and 6 are plotted for, and the design point of
# ``radial_inflow_turbine_blade_loading.py``.
CONE_ANGLE = np.radians(90.0)
BLADE_ANGLE = np.radians(0.0)

# Reference blade speed for the operating-point sweeps: only the flow
# coefficient Cm2/U2 = W_mer/U matters (mu, and hence the resulting
# incidence, depend only on that ratio here), so its absolute value is
# arbitrary.
U_REF = 300.0

BLADE_NUMBERS = np.arange(10, 26)
CORRELATIONS = ('chen', 'stanitz')


def solve_slip_and_incidence(
    num_blades: float, flow_coeff: float, correlation: str
) -> tuple[float, float]:
    """
    Build and solve the single-node, two-equation
    ``OptimalIncidenceRadialInflowTurbine`` system for one blade number,
    flow coefficient (Cm2/U2) and correlation choice.

    Returns
    -------
    (slip_factor, incidence_deg) : tuple[float, float]
        The slip factor mu (independent of the flow coefficient) and the
        resulting incidence angle beta2 - beta_b2, in degrees (which
        does depend on it).
    """
    system = CasadiSystem(num_span=1)
    system.add_equation(OptimalIncidenceRadialInflowTurbine(correlation=correlation), 0)
    system.add_boundary_conditions(
        {
            n0.geo.MeridionalAngle: CONE_ANGLE,
            n0.geo.MetalAngle: BLADE_ANGLE,
            n0.geo.NumBlades: num_blades,
            n0.kin.W_mer: flow_coeff * U_REF,
            n0.kin.BladeSpeed: U_REF,
        }
    )
    system.build()

    x0 = system.get_guess(
        {
            n0.kin.FlowAngleRel: np.radians(-20.0),
            n0.oth.SlipFactor: 0.85,
        }
    )
    kn = system.get_boundary_conds()

    rootfinder = system.make_rootfinder('kinsol')
    solution = solve_root_problem(rootfinder, x0, kn, suppress_output=True)
    sol_dict = system.sol_to_dict(solution)

    beta2 = float(sol_dict[n0.kin.FlowAngleRel][0])
    slip = float(sol_dict[n0.oth.SlipFactor][0])
    incidence_deg = np.degrees(beta2) - np.degrees(BLADE_ANGLE)
    return slip, incidence_deg


def solve_optimum_incidence(num_blades: float, correlation: str) -> tuple[float, float]:
    """
    Incidence at the Chen & Baines theoretical optimum-loading operating
    point (Eqs. 2, 5, 7a-7b, for the zero blade angle case here):
    Cm2/U2 = sqrt(mu * (1 - mu)).

    Since mu does not depend on the flow coefficient, this is a two-pass
    solve: first at an arbitrary flow coefficient just to read off mu,
    then again at the flow coefficient that mu implies is optimal, so
    the reported incidence matches the paper's design-point result.
    """
    slip, _ = solve_slip_and_incidence(
        num_blades, flow_coeff=0.3, correlation=correlation
    )
    phi_opt = float(np.sqrt(slip * (1 - slip)))
    return solve_slip_and_incidence(
        num_blades, flow_coeff=phi_opt, correlation=correlation
    )


if __name__ == '__main__':
    results = {
        correlation: np.array(
            [solve_optimum_incidence(float(z), correlation) for z in BLADE_NUMBERS]
        )
        for correlation in CORRELATIONS
    }

    table = [
        [
            z,
            *results['chen'][i],
            *results['stanitz'][i],
        ]
        for i, z in enumerate(BLADE_NUMBERS)
    ]
    logger.info(
        '\n'
        + tabulate(
            table,
            headers=[
                'Z',
                'mu (Chen Eq.8)',
                'i2 [deg] (Chen)',
                'mu (Stanitz Eq.4)',
                'i2 [deg] (Stanitz)',
            ],
            floatfmt='.3f',
        )
    )

    # Operating-point sweep at fixed Z: demonstrates that incidence is
    # NOT a fixed function of geometry alone -- it depends on how far
    # the actual flow coefficient departs from the optimum design value.
    Z_FIXED = 16.0
    FLOW_COEFFS = np.linspace(0.1, 0.5, 17)
    op_point_incidence = {
        correlation: np.array(
            [
                solve_slip_and_incidence(Z_FIXED, phi, correlation)[1]
                for phi in FLOW_COEFFS
            ]
        )
        for correlation in CORRELATIONS
    }

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    for correlation, style in zip(CORRELATIONS, ('-o', '--s')):
        slip, incidence = results[correlation].T
        axs[0].plot(BLADE_NUMBERS, slip, style, label=correlation)
        axs[1].plot(BLADE_NUMBERS, incidence, style, label=correlation)
        axs[2].plot(
            FLOW_COEFFS, op_point_incidence[correlation], style, label=correlation
        )

    # Reference band from the paper's Fig. 6: "the optimum incidence angle
    # for a radial turbine is widely reported to be about -20 deg"
    axs[1].axhline(
        -20.0, color='gray', linestyle=':', label='reported optimum (~-20 deg)'
    )

    axs[0].set_xlabel('Blade number Z')
    axs[0].set_ylabel(r'Inlet loading coefficient $\psi_2 = \mu$ (zero blade angle)')
    axs[0].set_title('Fig. 3 -- inlet loading vs. blade number')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].set_xlabel('Blade number Z')
    axs[1].set_ylabel('Incidence at the optimum operating point [deg]')
    axs[1].set_title('Fig. 6 -- optimum incidence vs. blade number')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].set_xlabel(r'Flow coefficient $C_{m2}/U_2$')
    axs[2].set_ylabel('Incidence [deg]')
    axs[2].set_title(f'Incidence vs. operating point (Z = {Z_FIXED:.0f})')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
