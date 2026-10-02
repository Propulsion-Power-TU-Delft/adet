"""
Meanline blade-to-blade loading of a radial impeller.

Implements, at the meanline only, the inviscid quasi-3D flow model of
Van den Braembussche, *Design and Analysis of Centrifugal Compressors*
(2019), Chapter 3, Section 3.1 "Inviscid Impeller Flow Calculation",
Eqs. (3.1)-(3.20). Plots the resulting suction-side (SS) / pressure-side
(PS) relative velocity and static pressure distribution along the
meanline of a prescribed impeller geometry.

Only the blade-to-blade part of the model (Eqs. 3.7, 3.13-3.15) is
implemented -- the meridional (hub-to-shroud) problem (Eqs. 3.1-3.6)
collapses to a single meanline value since there's no hub/shroud split
here. The general loading form (3.13) is used, not its axial/2D/
straight-channel special cases (3.16-3.18), since none apply to a real
curved rotating radial passage. The slip/deviation model (Eqs. 3.8-3.12)
is approximated by ``SlipTransition``: zero deviation up to a transition
point, then a quadratic blend to the trailing edge whose free coefficient
is set by the Kutta condition (zero blade loading at exit) rather than an
empirical slip correlation, since ADeT has none built in. Blade-surface
static pressure (not explicit in the book, but a direct consequence of
its equations) follows from rothalpy conservation (Eq. 1.68) plus the
inviscid/isentropic assumption shared by both surfaces.

The meanline is discretized into ``N_STATIONS`` nodes from the leading
edge (node 0) to the trailing edge. Standard ADeT equations (Kinematics,
AnnulusAreas, MassAreaRelation, ZeroBlockage, TotalStaticMatching,
ZeroDeviation, MassConservation, IsentropicLink) are reused unmodified;
only the physics specific to this chapter is added as custom equations
and variables.
"""

import logging

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
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
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.geometry import BezierCurve
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables, VariableEnum
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)
setup_logger(logger)

thrm = ThermoVariables()

# ============================================================
# 1. Impeller geometry and operating point (all prescribed)
# ============================================================
N_STATIONS = 20  # LE = station 0, TE = station N_STATIONS - 1

R_IN = 0.055  # [m] meanline radius at the leading edge (inducer)
R_OUT = 0.130  # [m] meanline radius at the trailing edge
Z_LENGTH = 0.090  # [m] axial extent of the meridional path (for plotting)

# Meridional flow (tangent) angle at each end (BezierCurve convention:
# 0 deg = axial +z, +90 deg = radial outward, -90 deg = radial inward).
# Angles are the physical direction of travel, not mirrored between ends
# -- getting the sign wrong makes the meanline radius overshoot
# non-monotonically (checked below).
MERID_ANGLE_IN = np.radians(0.0)  # [rad] purely axial at the inlet
MERID_ANGLE_OUT = np.radians(90.0)  # [rad] purely radial (outward) at the outlet

H_IN = 0.026  # [m] passage width (blade height) at the leading edge
H_OUT = 0.012  # [m] passage width (blade height) at the trailing edge

BETA_BL_IN = np.radians(-48.0)  # [rad] blade metal angle at the LE
BETA_BL_OUT = np.radians(-14.0)  # [rad] blade metal angle at the TE (backsweep)

NUM_BLADES = 22
BLADE_THICKNESS = 0.0015  # [m], constant along the streamline

RPM = 16_000.0
OMEGA = RPM * 2 * np.pi / 60  # [rad/s]

MASS_FLOW = 0.5  # [kg/s]
P0_TOT = 101_325.0  # [Pa]
T0_TOT = 293.15  # [K]

# ============================================================
# 2. Precompute the meanline geometry (purely descriptive, not
#    part of the nonlinear system: R, H and beta_bl are boundary
#    conditions of the equation system, not unknowns)
# ============================================================
_meridional_path = BezierCurve(
    z_in=0.0,
    z_out=Z_LENGTH,
    radius_in=R_IN,
    radius_out=R_OUT,
    angle_in=MERID_ANGLE_IN,
    angle_out=MERID_ANGLE_OUT,
    n_points=400,
)

if R_IN != R_OUT:
    _r_diffs = np.diff(_meridional_path.r_coords)
    _expected_sign = np.sign(R_OUT - R_IN)
    if not np.all(_expected_sign * _r_diffs >= -1e-9 * max(R_IN, R_OUT)):
        raise ValueError(
            'The meanline radius is not monotonic between R_IN and R_OUT for '
            f'the chosen MERID_ANGLE_IN={np.degrees(MERID_ANGLE_IN):.1f} deg / '
            f'MERID_ANGLE_OUT={np.degrees(MERID_ANGLE_OUT):.1f} deg: the Bezier '
            'tangent at one end points away from the other end, making the '
            'curve overshoot. Angles are given as the physical direction of '
            'travel (0=+z axial, +90=+r outward, -90=-r inward), not mirrored '
            'between inlet and outlet -- see the NOTE above MERID_ANGLE_IN.'
        )

_arc_length = np.concatenate(
    [
        [0.0],
        np.cumsum(
            np.hypot(
                np.diff(_meridional_path.z_coords),
                np.diff(_meridional_path.r_coords),
            )
        ),
    ]
)
S_MAX = _arc_length[-1]

_s_stations = np.linspace(0.0, S_MAX, N_STATIONS)
R_STATIONS = np.interp(_s_stations, _arc_length, _meridional_path.r_coords)
Z_STATIONS = np.interp(_s_stations, _arc_length, _meridional_path.z_coords)
H_STATIONS = np.linspace(H_IN, H_OUT, N_STATIONS)
BETA_BL_STATIONS = np.linspace(BETA_BL_IN, BETA_BL_OUT, N_STATIONS)
DELTA_S = np.diff(_s_stations)  # streamwise spacing of each interval

# ============================================================
# 3. Node and custom variable declarations
# ============================================================
nodes = [NodeVariables(i) for i in range(N_STATIONS)]

# Local node placeholders (upstream/downstream) for residual() hints,
# unrelated to absolute station index -- the standard n0/n1 convention.
n0 = NodeVariables(0)
n1 = NodeVariables(1)


class BladeLoadingVariables(VariableEnum):
    """
    Custom variables for the blade-to-blade loading (Section 3.1.2) that
    are not part of ADeT's built-in variable library.
    """

    W_ss = VarSpec('W_ss', 'm / s', 120.0, (0.0, 2e3))
    """Relative velocity on the suction side, :math:`W_{SS}`."""

    W_ps = VarSpec('W_ps', 'm / s', 80.0, (0.0, 2e3))
    """Relative velocity on the pressure side, :math:`W_{PS}`."""

    DeltaW = VarSpec('delta_W', 'm / s', 20.0, (-1e3, 1e3))
    """Blade loading :math:`W_{SS} - W_{PS}`, Eq. (3.13)."""

    P_ss = VarSpec('p_ss', 'Pa', 8e4, (1e2, 2e7))
    """Static pressure on the suction side."""

    P_ps = VarSpec('p_ps', 'Pa', 1.2e5, (1e2, 2e7))
    """Static pressure on the pressure side."""


bl_nodes = [BladeLoadingVariables(i) for i in range(N_STATIONS)]
bl0 = BladeLoadingVariables(0)
bl1 = BladeLoadingVariables(1)


# ============================================================
# 4. Custom equations specific to this book chapter
# ============================================================
class RothalpyConservation(EquationBase):
    """Conservation of rothalpy between two consecutive meanline stations
    (Van den Braembussche Eq. 1.68):

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


class BladeToBladeLoading(EquationBase):
    """General blade-to-blade suction-to-pressure side velocity
    difference (Van den Braembussche Eq. 3.13):

    .. math::
        W_{SS} - W_{PS} = \\left(\\frac{2\\pi}{Z_r} -
        \\frac{\\delta_{bl}}{R \\cos\\beta_{fl}}\\right)
        \\frac{d}{ds}\\left(\\Omega R^2 -
        \\widetilde{W}_m R \\tan\\beta_{fl}\\right)

    The streamwise derivative is a backward finite difference between the
    upstream (node 0) and downstream (node 1) stations (``delta_s``).
    """

    config = EquationConfig(manual_units=('m / s',))

    def __init__(self, delta_s: float, **kwargs):
        super().__init__(**kwargs)
        self.delta_s = delta_s

    def residual(
        self,
        omega0: n0.kin.Omega.Hint,
        r0: n0.geo.RDistr.Hint,
        wm0: n0.kin.W_mer.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        omega1: n1.kin.Omega.Hint,
        r1: n1.geo.RDistr.Hint,
        wm1: n1.kin.W_mer.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        z_r1: n1.geo.NumBlades.Hint,
        delta_bl1: n1.geo.BldThick.Hint,
        delta_w1: bl1.DeltaW.Hint,
    ):
        rotation_term = (omega1 * r1**2 - omega0 * r0**2) / self.delta_s  # Eq. (3.14)
        turning_term = (
            wm1 * r1 * np.tan(beta1) - wm0 * r0 * np.tan(beta0)
        ) / self.delta_s  # Eq. (3.15)

        pitch_minus_thickness = 2 * np.pi / z_r1 - delta_bl1 / (r1 * np.cos(beta1))

        loading = pitch_minus_thickness * (rotation_term - turning_term)

        return delta_w1 - loading


class SlipTransition(EquationBase):
    """Simplified stand-in for the flow-angle/blade-angle deviation model
    of Van den Braembussche Eqs. (3.9)-(3.12), used approaching the
    trailing edge where the loading must relax to zero (Kutta condition).

    Downstream of a transition point :math:`s^*`, the flow angle follows
    a quadratic blend continuous in value and slope with the blade metal
    angle at :math:`s^*` (Eqs. 3.9-3.11):

    .. math::
        \\beta_{fl}(s) = A (s - s^*)^2 +
        \\left.\\frac{d\\beta_{bl}}{ds}\\right|_{s^*} (s - s^*) +
        \\beta_{bl}(s^*)

    Rather than fixing :math:`A` from an empirical slip correlation
    (Eq. 3.12, absent in ADeT), the trailing-edge angle is left free and
    determined by the Kutta condition elsewhere.
    """

    config = EquationConfig(manual_units=('rad',))

    def __init__(
        self,
        s_i: float,
        s_star: float,
        s_end: float,
        beta_bl_star: float,
        slope_star: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.s_i = s_i
        self.s_star = s_star
        self.s_end = s_end
        self.beta_bl_star = beta_bl_star
        self.slope_star = slope_star

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        beta_slip1: n1.kin.FlowAngleRel.Hint,
    ):
        span = self.s_end - self.s_star
        a_coeff = (beta_slip1 - self.beta_bl_star - self.slope_star * span) / span**2

        beta_target = (
            a_coeff * (self.s_i - self.s_star) ** 2
            + self.slope_star * (self.s_i - self.s_star)
            + self.beta_bl_star
        )
        return beta0 - beta_target


class SuctionPressureVelocities(EquationBase):
    """Superpose the blade-to-blade loading on the pitchwise-averaged
    relative velocity to recover SS/PS velocities (Van den Braembussche,
    text below Eq. 3.13)."""

    def residual(
        self,
        w_mean0: n0.kin.W_mag.Hint,
        delta_w0: bl0.DeltaW.Hint,
        w_ss0: bl0.W_ss.Hint,
        w_ps0: bl0.W_ps.Hint,
    ):
        r1 = w_ss0 - (w_mean0 + delta_w0 / 2)
        r2 = w_ps0 - (w_mean0 - delta_w0 / 2)
        return r1, r2


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
_BETA_BL_SLOPE = (BETA_BL_OUT - BETA_BL_IN) / S_MAX  # constant (linear law)

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
    # At TRAILING_EDGE the flow angle is left free, pinned by the Kutta
    # condition instead (see BOUNDARY_CONDITIONS).
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

# Kutta condition at the TE: W_SS = W_PS (Van den Braembussche Sec 3.1.3).
BOUNDARY_CONDITIONS[bl_nodes[TRAILING_EDGE].DeltaW] = 0.0

system.add_boundary_conditions(BOUNDARY_CONDITIONS)

# ============================================================
# 7. Build and solve
# ============================================================
system.build()

x0 = system.get_guess(fallback=0.6)
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

# Diagnostic for Eqs. (3.19)-(3.20): R*Vu is only constant for a very
# specific "force-free" backsweep law. A generic backswept blade like
# this one loads the flow, i.e. d(R*Vu)/ds != 0.
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

# --- Meridional profile with station markers
axs[0].plot(_meridional_path.z_coords, _meridional_path.r_coords, 'k-')
axs[0].plot(Z_STATIONS, R_STATIONS, 'o', color='tab:red')
axs[0].set_aspect('equal')
axs[0].set_xlabel(r'$z$ / [m]')
axs[0].set_ylabel(r'$r$ / [m]')
axs[0].set_title('Meanline meridional path')
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
