"""
Meanline blade-to-blade loading of a fully radial, straight-vane rotor.

Variant of ``radial_impeller_blade_loading.py``: purely radial meanline, constant
blade height, straight (zero blade angle) radial vanes.
"""

import logging

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.equations.base_equation import EquationBase, EquationConfig
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.throughflow import (
    BladeLoadingVariables,
    BladeToBladeLoading,
    SlipTransition,
    SuctionPressureVelocities,
)
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

thrm = ThermoVariables()

# ============================================================
# 1. Rotor geometry and operating point (all prescribed)
# ============================================================
N_STATIONS = 15  # LE = station 0, TE = station N_STATIONS - 1

R_IN = 0.050  # [m] meanline radius at the leading edge
R_OUT = 0.110  # [m] meanline radius at the trailing edge

H_CONST = 0.020  # [m] passage width (blade height), constant along the vane

BETA_BL_DEG = -20.0
BETA_BL = np.radians(BETA_BL_DEG)  # [rad]; 0 = straight radial vane (Fig. 3.8a)

NUM_BLADES = 19
BLADE_THICKNESS = 0.0015  # [m], constant along the streamline

RPM = 5_000.0
OMEGA = RPM * 2 * np.pi / 60  # [rad/s]

MASS_FLOW = 0.5  # [kg/s]
P0_TOT = 101_325.0  # [Pa]
T0_TOT = 293.15  # [K]

# ============================================================
# 2. Precompute the meanline geometry (R, H, beta_bl are boundary
#    conditions, not unknowns)
# ============================================================
# Purely radial meanline (no axial-to-radial bend): s coincides exactly
# with the radius, s = R - R_IN.
_s_stations = np.linspace(0.0, R_OUT - R_IN, N_STATIONS)
S_MAX = _s_stations[-1]

R_STATIONS = R_IN + _s_stations
H_STATIONS = np.full(N_STATIONS, H_CONST)
BETA_BL_STATIONS = np.full(N_STATIONS, BETA_BL)
DELTA_S = np.diff(_s_stations)  # streamwise spacing of each interval

# Blade-to-blade-plane shape traced by the vane, for plotting only: with a
# purely radial meridional coordinate (dm = dR), tan(beta_bl) = R * dtheta/dR,
# so the vane's angular coordinate is the running integral below (zero for
# a straight radial vane, a logarithmic spiral otherwise).
_dtheta_dR = np.tan(BETA_BL_STATIONS) / R_STATIONS
_avg_dtheta_dR = 0.5 * (_dtheta_dR[1:] + _dtheta_dR[:-1])
_dtheta = _avg_dtheta_dR * np.diff(R_STATIONS)
THETA_STATIONS = np.concatenate([[0.0], np.cumsum(_dtheta)])

# ============================================================
# 3. Node and custom variable declarations
# ============================================================
nodes = [NodeVariables(i) for i in range(N_STATIONS)]

n0 = NodeVariables(0)
n1 = NodeVariables(1)

bl_nodes = [BladeLoadingVariables(i) for i in range(N_STATIONS)]
bl0 = BladeLoadingVariables(0)
bl1 = BladeLoadingVariables(1)


# ============================================================
# 4. Custom equations specific to this test
# ============================================================
class RothalpyConservation(EquationBase):
    """Conservation of rothalpy between two consecutive meanline stations
    (Eq. 1.68):

    .. math::
        I = h_{t}^{rel} - \\frac{U^2}{2} = \\text{const.}
    """

    def residual(
        self,
        h_rel0: n0.rlt.Enthalpy.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        h_rel1: n1.rlt.Enthalpy.Hint,
        u1: n1.kin.BladeSpeed.Hint,
    ):
        rothalpy0 = h_rel0 - u0**2 / 2
        rothalpy1 = h_rel1 - u1**2 / 2
        return rothalpy1 - rothalpy0


class BladeSurfacePressures(EquationBase):
    """Static pressure on the suction (SS) and pressure (PS) blade
    surfaces, from rothalpy conservation (:class:`RothalpyConservation`)
    plus the shared meanline static entropy (inviscid assumption)."""

    config = EquationConfig(
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(thrm.Pressure,),
    )

    def residual(
        self,
        w_ss0: bl0.W_ss.Hint,
        w_ps0: bl0.W_ps.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        h_rel0: n0.rlt.Enthalpy.Hint,
        s0: n0.stc.Entropy.Hint,
        p_ss0: bl0.P_ss.Hint,
        p_ps0: bl0.P_ps.Hint,
    ):
        rothalpy = h_rel0 - u0**2 / 2

        h_ss = rothalpy - w_ss0**2 / 2 + u0**2 / 2
        h_ps = rothalpy - w_ps0**2 / 2 + u0**2 / 2

        p_ss_eos = self.eos(h_ss, s0)
        p_ps_eos = self.eos(h_ps, s0)

        return p_ss0 - p_ss_eos, p_ps0 - p_ps_eos


# ============================================================
# 5. Assemble the system
# ============================================================
system = CasadiSystem(num_span=1)

system.fluid_settings = FluidSettings(
    fluid_state=IdealGasState(gamma=1.4, gas_constant=287.0, viscosity=1.8e-5),
    update_variables=(thrm.Pressure, thrm.Temperature),
)

TRAILING_EDGE = N_STATIONS - 1

# Transition point s* (Eqs. 3.9-3.12): ZeroDeviation up to here, then
# SlipTransition's quadratic blend down to zero loading at the TE.
TRANSITION_STAR = int(round(0.6 * TRAILING_EDGE))
_BETA_BL_SLOPE = 0.0  # straight vane: blade angle is constant (zero slope)

EQUATIONS = {}
for i in range(N_STATIONS):
    EQUATIONS[Kinematics()] = i
    if i <= TRANSITION_STAR:
        EQUATIONS[ZeroDeviation()] = i
    elif i != TRAILING_EDGE:
        EQUATIONS[
            SlipTransition(
                s_i=_s_stations[i],
                s_star=_s_stations[TRANSITION_STAR],
                s_end=S_MAX,
                beta_bl_star=BETA_BL_STATIONS[TRANSITION_STAR],
                slope_star=_BETA_BL_SLOPE,
            )
        ] = (i, TRAILING_EDGE)
    EQUATIONS[AnnulusAreas()] = i
    EQUATIONS[ZeroBlockage()] = i
    EQUATIONS[MassAreaRelation()] = i
    EQUATIONS[TotalStaticMatching()] = i
    EQUATIONS[SuctionPressureVelocities()] = i
    EQUATIONS[BladeSurfacePressures()] = i

for i in range(1, N_STATIONS):
    EQUATIONS[MassConservation()] = (i - 1, i)
    EQUATIONS[IsentropicLink()] = (i - 1, i)
    EQUATIONS[RothalpyConservation()] = (i - 1, i)
    EQUATIONS[BladeToBladeLoading(delta_s=DELTA_S[i - 1])] = (i - 1, i)

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

# ============================================================
# 6. Boundary conditions
# ============================================================
BOUNDARY_CONDITIONS = {}
for i in range(N_STATIONS):
    n = nodes[i]
    BOUNDARY_CONDITIONS[n.geo.RDistr] = R_STATIONS[i]
    BOUNDARY_CONDITIONS[n.geo.HDistr] = H_STATIONS[i]
    BOUNDARY_CONDITIONS[n.geo.MetalAngle] = BETA_BL_STATIONS[i]
    BOUNDARY_CONDITIONS[n.geo.NumBlades] = NUM_BLADES
    BOUNDARY_CONDITIONS[n.geo.BldThick] = BLADE_THICKNESS
    BOUNDARY_CONDITIONS[n.kin.Omega] = OMEGA

# Inlet total thermodynamic state and target mass flow (node 0 only)
BOUNDARY_CONDITIONS[nodes[0].tot.Pressure] = Quantity(P0_TOT, 'Pa')
BOUNDARY_CONDITIONS[nodes[0].tot.Temperature] = Quantity(T0_TOT, 'K')
BOUNDARY_CONDITIONS[nodes[0].oth.StreamMassFlow] = MASS_FLOW

# Zero incidence at the LE: no blade loading yet.
BOUNDARY_CONDITIONS[bl_nodes[0].DeltaW] = 0.0

# Kutta condition at the TE: W_SS = W_PS (Section 3.1.3).
BOUNDARY_CONDITIONS[bl_nodes[TRAILING_EDGE].DeltaW] = 0.0

system.add_boundary_conditions(BOUNDARY_CONDITIONS)

# ============================================================
# 7. Build and solve
# ============================================================
system.build()

manual_guess = {}
for i in range(N_STATIONS):
    manual_guess[bl_nodes[i].P_ss] = 8e4
    manual_guess[bl_nodes[i].P_ps] = 1.2e5

x0 = system.get_guess(manual_guess, fallback=0.6)
kn = system.get_boundary_conds()
bnd = system.get_bounds()

rootfinder = system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.tol': 1e-8,
        'ipopt.max_iter': 500,
    },
)
solution = solve_root_problem(rootfinder, x0, kn, bnd, suppress_output=False)

# Polish with a Newton-Krylov solve
rootfinder_kinsol = system.make_rootfinder('kinsol')
solution = solve_root_problem(rootfinder_kinsol, solution, kn)

sol_dict = system.sol_to_dict(solution)

# ============================================================
# 8. Post-processing and plots
# ============================================================
s_over_smax = _s_stations / S_MAX

w_mean = np.array([sol_dict[nodes[i].kin.W_mag][0] for i in range(N_STATIONS)])
w_ss = np.array([sol_dict[bl_nodes[i].W_ss][0] for i in range(N_STATIONS)])
w_ps = np.array([sol_dict[bl_nodes[i].W_ps][0] for i in range(N_STATIONS)])
p_ss = np.array([sol_dict[bl_nodes[i].P_ss][0] for i in range(N_STATIONS)])
p_ps = np.array([sol_dict[bl_nodes[i].P_ps][0] for i in range(N_STATIONS)])
p_mean = np.array([sol_dict[nodes[i].stc.Pressure][0] for i in range(N_STATIONS)])

# Diagnostic for Eqs. (3.19)-(3.20): even with zero blade curvature,
# d(R*Vu)/ds = 2*Omega*R*dR/ds != 0 for a rotating radial channel -- a
# straight radial vane still does work on the flow. Eq. (3.19)'s
# "force-free" condition would require a specific *backward* curvature
# (a non-radial, non-straight vane) to null this, which is exactly why
# real radial impellers are backswept rather than straight.
r_vu = np.array(
    [
        sol_dict[nodes[i].geo.RDistr][0] * sol_dict[nodes[i].kin.V_tan][0]
        for i in range(N_STATIONS)
    ]
)
logger.info(f'R*Vu along the meanline (Eq. 3.20 diagnostic): {r_vu}')
logger.info(
    f'Euler work per unit mass (Omega * d(R*Vu)): '
    f'{OMEGA * (r_vu[-1] - r_vu[0]):.1f} J/kg'
)

fig, axs = plt.subplots(1, 3, figsize=(16, 5))

# --- Top (blade-to-blade plane) view of all NUM_BLADES vanes, traced from
# THETA_STATIONS (see above) so this always reflects BETA_BL, however it
# is set: a straight radial line for BETA_BL = 0, a logarithmic spiral
# otherwise. There is no meridional bend to plot here (unlike the
# backswept-impeller example): the rotor is flat, so this is the more
# informative geometry view for this case.
for theta0 in np.linspace(0.0, 2 * np.pi, NUM_BLADES, endpoint=False):
    theta_vane = theta0 + THETA_STATIONS
    axs[0].plot(
        R_STATIONS * np.cos(theta_vane),
        R_STATIONS * np.sin(theta_vane),
        'k-',
    )
axs[0].add_patch(Circle((0, 0), R_IN, fill=False, linestyle='--', color='gray'))
axs[0].add_patch(Circle((0, 0), R_OUT, fill=False, linestyle='--', color='gray'))
axs[0].set_xlim(-1.15 * R_OUT, 1.15 * R_OUT)
axs[0].set_ylim(-1.15 * R_OUT, 1.15 * R_OUT)
axs[0].set_aspect('equal')
axs[0].set_xlabel(r'$x$ / [m]')
axs[0].set_ylabel(r'$y$ / [m]')
axs[0].set_title(
    f'Radial vanes (top view, {NUM_BLADES} shown, '
    f'$\\beta_{{bl}}$={np.degrees(BETA_BL):.0f}deg)'
)
axs[0].grid(True, alpha=0.3)

# --- Velocity distribution (mirrors Figure 3.2 / 3.12 of the book)
axs[1].plot(s_over_smax, w_ss, '-o', label=r'$W_{SS}$')
axs[1].plot(s_over_smax, w_mean, '--', color='k', label=r'$\widetilde{W}$')
axs[1].plot(s_over_smax, w_ps, '-o', label=r'$W_{PS}$')
axs[1].set_xlabel(r'$s / s_{max}$')
axs[1].set_ylabel(r'$W$ / [m/s]')
axs[1].set_title('Blade-to-blade velocity distribution')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# --- Pressure distribution
axs[2].plot(s_over_smax, p_ss / 1e5, '-o', label=r'$p_{SS}$')
axs[2].plot(s_over_smax, p_mean / 1e5, '--', color='k', label=r'$p$ (mean)')
axs[2].plot(s_over_smax, p_ps / 1e5, '-o', label=r'$p_{PS}$')
axs[2].set_xlabel(r'$s / s_{max}$')
axs[2].set_ylabel(r'$p$ / [bar]')
axs[2].set_title('Blade surface pressure distribution')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
