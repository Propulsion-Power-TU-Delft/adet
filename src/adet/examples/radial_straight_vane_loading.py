"""
Meanline blade-to-blade loading of a fully radial, straight-vane rotor.

This is a variant of ``radial_impeller_blade_loading.py`` (read that one
first) for the specific geometry of Section 3.1.2's "2D radial rotating
channel" (Figure 3.8a): a rotor with purely radial meanline (no axial-to-
radial bend), constant blade height, and straight (uncambered, zero blade
angle) radial vanes -- e.g. a simple radial-vaned blower/pump impeller,
as opposed to the backswept centrifugal-compressor impeller of the other
example.

The point of this variant: the book presents Eq. (3.18) as a *separate*,
simplified formula for this geometry,

.. math::
    W_{SS} - W_{PS} = \\left(\\frac{2\\pi R}{Z_r} -
    \\frac{\\delta_{bl}}{\\cos\\beta}\\right) 2 \\Omega \\frac{dR}{ds}

obtained from the general Eq. (3.13) by assuming :math:`\\widetilde{W}_m
R = \\text{const.}` and constant :math:`\\beta`. We do *not* implement a
separate equation for it: with straight radial vanes, :math:`\\beta_{fl}
= \\beta_{bl} = 0` everywhere (away from the trailing-edge transition
discussed below), so :math:`\\tan\\beta_{fl} \\equiv 0` and the "turning"
term of the already-implemented general :class:`BladeToBladeLoading`
(Eq. 3.15, :math:`d(\\widetilde{W}_m R \\tan\\beta_{fl})/ds`) vanishes
identically. What is left is exactly the "rotation" term (Eq. 3.14,
:math:`d(\\Omega R^2)/ds`), i.e. precisely Eq. (3.18) -- our finite
difference uses the exact :math:`\\Omega(R_1^2 - R_0^2)/\\Delta s` rather
than the book's linearization :math:`2 \\Omega R\\, dR/ds`, which agree in
the continuum limit. This is a genuine consequence of the model, not a
coincidence: it demonstrates that Eq. (3.13) is the general statement and
Eqs. (3.16)-(3.18) are just special cases of it for particular geometries.

All the other modeling choices are identical to
``radial_impeller_blade_loading.py``: meanline only (no hub/shroud split,
so Eqs. 3.1-3.6 don't apply), the Kutta condition enforced at the trailing
edge via the same simplified slip-transition device (:class:`SlipTransition`,
a stand-in for Eqs. 3.9-3.12), rothalpy conservation (Eq. 1.68) plus the
isentropic assumption to get the SS/PS static pressure, and the SS/PS
velocity superposition rule below Eq. (3.13). See that file's module
docstring for the full rationale; it is not repeated here.
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
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables, VariableEnum
from adet.varspec import VarSpec

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

BETA_BL_DEG = -0.0
BETA_BL = np.radians(BETA_BL_DEG)  # [rad]; 0 = straight radial vane (Fig. 3.8a)

NUM_BLADES = 9
BLADE_THICKNESS = 0.0015  # [m], constant along the streamline

RPM = 5_000.0
OMEGA = RPM * 2 * np.pi / 60  # [rad/s]

MASS_FLOW = 0.5  # [kg/s]
P0_TOT = 101_325.0  # [Pa]
T0_TOT = 293.15  # [K]

# ============================================================
# 2. Precompute the meanline geometry (purely descriptive, not
#    part of the nonlinear system: R, H and beta_bl are boundary
#    conditions of the equation system, not unknowns)
# ============================================================
# The meanline is purely radial (no axial-to-radial bend at all, unlike
# the backswept-impeller example): the streamwise coordinate s coincides
# exactly with the radius, s = R - R_IN.
_s_stations = np.linspace(0.0, R_OUT - R_IN, N_STATIONS)
S_MAX = _s_stations[-1]

R_STATIONS = R_IN + _s_stations
H_STATIONS = np.full(N_STATIONS, H_CONST)
BETA_BL_STATIONS = np.full(N_STATIONS, BETA_BL)
DELTA_S = np.diff(_s_stations)  # streamwise spacing of each interval

# Blade-to-blade-plane shape traced by the vane, for plotting only: with a
# purely radial meridional coordinate (dm = dR), the blade angle satisfies
# tan(beta_bl) = R * dtheta/dR, so the vane's angular coordinate is the
# running integral below. BETA_BL = 0 (a truly *straight* radial vane)
# makes this identically zero; a nonzero (constant) blade angle traces an
# equiangular (logarithmic) spiral instead -- Figure 3.8b's "backward
# curved rotating channel" rather than Figure 3.8a's straight one. This is
# recomputed from BETA_BL_STATIONS so the plot always matches whatever
# blade angle is set above, even if it is changed to a nonzero or
# station-varying value.
_dtheta_dR = np.tan(BETA_BL_STATIONS) / R_STATIONS
_avg_dtheta_dR = 0.5 * (_dtheta_dR[1:] + _dtheta_dR[:-1])
_dtheta = _avg_dtheta_dR * np.diff(R_STATIONS)
THETA_STATIONS = np.concatenate([[0.0], np.cumsum(_dtheta)])

# ============================================================
# 3. Node and custom variable declarations
# ============================================================
nodes = [NodeVariables(i) for i in range(N_STATIONS)]

# Local node placeholders 0 ("upstream"/"self") and 1 ("downstream"/"other"),
# used only inside the equation classes' residual() hints below -- exactly
# the n0/n1 convention used throughout adet.equations.* and the other
# examples. These are unrelated to the *absolute* station index: ADeT remaps
# them to whichever absolute nodes are passed to `add_equation(eq, pos)`.
# (Indexing directly into `nodes` inside a type hint, e.g. `nodes[0].kin...`,
# would work at runtime but defeats static type checkers, since `nodes[0]`
# is not a statically resolvable expression.)
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
    """
    Conservation of rothalpy between two consecutive meanline stations.

    .. math::
        I = h + \\frac{W^2}{2} - \\Omega R^2 \\cdot 0
             \\quad\\Rightarrow\\quad
        I = h_{t}^{rel} - \\frac{U^2}{2} = \\text{const.}

    This is Eq. (1.68) of the book, invoked in the text right after
    Eq. (3.2) when it is assumed that the rothalpy is uniform at the
    inlet and therefore constant everywhere in the impeller. ADeT's
    ``TotalStaticMatching`` equation already provides the relative total
    enthalpy :math:`h_t^{rel} = h + W^2/2`; here we only need to remove
    the blade speed contribution :math:`U^2/2` and match it between the
    two nodes.
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
    """
    General blade-to-blade suction-to-pressure side velocity difference,
    Eq. (3.13):

    .. math::
        W_{SS} - W_{PS} = \\left(\\frac{2\\pi}{Z_r} -
        \\frac{\\delta_{bl}}{R \\cos\\beta_{fl}}\\right)
        \\frac{d}{ds}\\left(\\Omega R^2 -
        \\widetilde{W}_m R \\tan\\beta_{fl}\\right)

    The streamwise derivative is evaluated with a backward finite
    difference between the upstream (node 0) and downstream (node 1)
    meanline stations, in the same spirit as the streamline-curvature
    recurrence of Eq. (3.5). The two contributions to the derivative are
    kept separate to mirror Eqs. (3.14) (rotational term,
    :math:`d(\\Omega R^2)/ds`) and (3.15) (blade-to-blade turning term,
    :math:`d(\\widetilde{W}_m R \\tan\\beta_{fl})/ds`).

    Parameters
    ----------
    delta_s : float
        Streamwise (arc length) distance between the two meanline
        stations linked by this equation instance, :math:`\\Delta s`.
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
    """
    Simplified stand-in for the flow-angle/blade-angle deviation model of
    Eqs. (3.9)-(3.12), used only over the small region approaching the
    trailing edge where the loading must relax to zero to satisfy the
    Kutta condition (see the ``DeltaW`` boundary condition at the
    trailing edge, further below).

    Downstream of a transition point :math:`s^*`, the flow angle is
    approximated by a second-degree polynomial in the streamwise
    coordinate (Eq. 3.9), with two of its three coefficients fixed by
    continuity of the flow angle and of its slope with the blade metal
    angle at :math:`s^*` (Eqs. 3.10, 3.11):

    .. math::
        \\beta_{fl}(s) = A (s - s^*)^2 +
        \\left.\\frac{d\\beta_{bl}}{ds}\\right|_{s^*} (s - s^*) +
        \\beta_{bl}(s^*)

    The book fixes the remaining coefficient :math:`A` from a prescribed
    trailing-edge slip angle (Eq. 3.12, :math:`\\beta_{2,fl} =
    \\beta_{2,slip}`), obtained from an empirical slip correlation. Since
    ADeT has no slip correlation built in, and this example wants to
    *enforce* the Kutta condition directly rather than reproduce a
    particular slip factor, :math:`\\beta_{2,slip}` is used the other way
    round here: it is simply the (free) flow angle at the trailing edge,
    which the Kutta condition (elsewhere) determines self-consistently.
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
    """
    Superpose the blade-to-blade loading on the pitchwise-averaged
    relative velocity to recover the SS and PS velocity distribution, as
    described just below Eq. (3.13): "Superposing this velocity
    difference on the pitchwise averaged value :math:`\\widetilde{W}`
    provides the SS and PS velocity distribution."
    """

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
    """
    Static pressure on the suction (SS) and pressure (PS) blade surfaces.

    Not an explicit equation of Section 3.1, but the direct consequence
    of combining conservation of rothalpy (Eq. 1.68, see
    :class:`RothalpyConservation`) with the isentropic (inviscid)
    assumption of Section 3.1 -- both surfaces share the meanline static
    entropy -- to get the local static enthalpy on each surface from its
    local relative velocity, and then the equation of state to convert
    each (h, s) pair into a pressure.
    """

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

# Station index of the transition point s* (Eqs. 3.9-3.12): upstream of
# it the flow follows the blade angle exactly (ZeroDeviation); from it to
# the trailing edge, the flow angle is left free and instead follows the
# smooth quadratic blend of SlipTransition, which relaxes the loading to
# zero by the trailing edge (Kutta condition).
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
    # At i == TRAILING_EDGE, the flow angle is left free entirely: it *is*
    # the beta_2_slip referenced by SlipTransition above, and it is
    # itself pinned by the Kutta condition (see BOUNDARY_CONDITIONS).
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

# No blade loading right at the leading edge: the blade angle is assumed
# to match the relative flow angle exactly at inlet (zero incidence), so
# the loading described by Eq. (3.13) only builds up downstream of it.
BOUNDARY_CONDITIONS[bl_nodes[0].DeltaW] = 0.0

# Kutta condition at the trailing edge: W_SS = W_PS, i.e. zero blade
# loading right at the exit (Section 3.1.3, "The zero velocity difference
# at the trailing edge is in agreement with the Kutta conditions"). Since
# ZeroDeviation was not added at the trailing edge above, the flow angle
# there is free to deviate from the blade metal angle -- exactly the
# mechanism the book attributes this condition to -- and BladeToBladeLoading
# now determines that deviation instead of determining DeltaW.
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
