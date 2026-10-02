"""
Meridional (hub-to-shroud) velocity field of a radial impeller.

Implements, via ADeT's equation-based system-building logic, Van den
Braembussche, *Design and Analysis of Centrifugal Compressors* (2019),
Section 3.1.1 "Meridional Velocity Calculation", full Eq. (3.3) -- not
its ``beta = 0`` special case (Eq. 3.4). The flow angle ``beta`` is a
*prescribed* input field here -- hub/shroud values given at the inlet
and at the outlet, each linearly distributed across the span, then
used as the streamwise endpoints of a parabola at each spanwise
station -- not solved for, consistent with the book's own model, where
the blade-to-blade split (Section 3.1.2, implemented separately in
``radial_impeller_blade_loading.py``) is what would actually determine
it from the blade geometry. See :func:`parabolic_beta_field`.

Equations
---------
Eq. (3.3), the hub-to-shroud force balance along a quasi-orthogonal
(:math:`n` runs from hub to shroud), **with the Coriolis term's sign
corrected** relative to how it is printed in the book:

.. math::
    \\frac{\\partial \\widetilde{W}}{\\partial n} =
    \\widetilde{W}\\frac{\\cos^2\\beta}{\\mathfrak{R}_n} -
    \\cos\\gamma \\sin\\beta
    \\left(2\\Omega + \\frac{\\widetilde{W}\\sin\\beta}{R}\\right)

The book's Eq. (3.3) has ``+cos(gamma)*sin(beta)*(2*Omega - ...)``, i.e.
the opposite sign on the ``2*Omega`` term. Re-deriving the normal-to-
streamline special case directly from Katsanis (1964) -- the paper the
book itself cites for this equation (``docs/Katsanis1964.pdf``, Eqs.
B10-B16, specialized to ``s = n``: ``dr/ds = cos(alpha)``,
``dz/ds = -sin(alpha)``, uniform rothalpy/prerotation across span) gives

.. math::
    \\frac{dW}{dn} = \\frac{W\\cos^2\\beta}{r_c} -
    \\cos\\alpha \\sin\\beta \\left(2\\omega + \\frac{W\\sin\\beta}{r}\\right)

(Katsanis's :math:`\\alpha` is the book's :math:`\\gamma`.) The
curvature and swirl terms match the book exactly; only the Coriolis
term's sign differs, which rules out a global convention difference
(e.g. a flipped :math:`n` direction would flip all three terms, not
one) -- it looks like a sign slip in the book's own eq. (3.3), not in
this transcription of it. Implemented (trapezoidal, between two
neighboring spanwise grid points) as :class:`MeridionalMomentumMarch`.

Eq. (3.6), mass conservation across a quasi-orthogonal. Rather than a
single sum, it is encoded as a spanwise *chain* of cumulative-flow
equations (:class:`MassFluxAccumulation`, trapezoidal), with the
cumulative flow pinned to 0 at the hub and to :math:`\\dot m` at the
shroud -- the ADeT-idiomatic way to express an integral constraint as
per-interval node equations (cf. ``nozzle_shock.py``).

Local density (:class:`LocalThermoState`) follows from rothalpy
conservation (Eq. 1.68), :math:`h = I_0 + U^2/2 - \\widetilde{W}^2/2`,
and the ideal-gas EOS at constant entropy (isentropic assumption),
:math:`\\rho = \\rho_{EOS}(h, s_{in})`, evaluated with ADeT's own
``adet.fluid.ideal_eos.IdealGasState`` (called directly, as a plain
(h, s) -> rho callable -- see the note below). Because the whole
system (every node, every equation) is solved simultaneously and
implicitly, the manual iterative procedure the book describes for the
compressible case is unnecessary here -- it's just what the nonlinear
solver is doing internally.

``IdealGasState`` is used directly (``.update()`` + ``.rhomass()``)
rather than through ADeT's ``FluidSettings``/``CasadiSystem`` EOS
wiring: that wiring declares pressure and temperature at every static/
total/relative-total state of every node as soon as ``fluid_settings``
is set on the system (it assumes the usual meanline P/T bookkeeping),
which would add thousands of unused variables/equations to this
span-only, custom problem. ``IdealGasState`` itself is fluid-model
agnostic about *how* it's called, so this still goes through the same
symbolic (sympy-derived, CasADi-compatible) solution the rest of ADeT
uses -- only the surrounding per-node bookkeeping is skipped.

Hypotheses and assumptions
---------------------------
* :math:`\\gamma` (local meridional streamline inclination) and
  :math:`\\mathfrak{R}_n` (meridional curvature) are purely geometric,
  computed once from the prescribed hub/shroud contours and held fixed.
* :math:`\\beta` follows a prescribed field (a stand-in for whatever the
  blade-to-blade solution would otherwise supply): linear hub-to-shroud
  interpolation of the given inlet value and of the given outlet value,
  then a streamwise parabola (with a spanwise-uniform mid-passage
  bulge) between those two local values at each spanwise station.
* Rothalpy and entropy are uniform, global constants, fixed by the
  inlet total conditions with no inlet prewhirl assumed (so that
  :math:`I_0 = h_{0,in}`), per the book's own simplifying assumption
  that rothalpy is uniform at the inlet and thus constant everywhere
  (text below Eq. 3.2).
* Ideal-gas air (:math:`\\gamma = 1.4`, :math:`R_{gas} = 287.05` J/(kg K)),
  via ``adet.fluid.ideal_eos.IdealGasState``.
* Quasi-orthogonals connecting hub and shroud are straight lines
  (Figure 3.5), and the intermediate "quasi streamsurfaces" are
  obtained by linear interpolation between the hub and shroud
  meridional contours, exactly as described below Eq. (3.5).
* Hub and shroud contours are generated as cubic Bezier splines (4
  control points: start, 2 intermediate, end -- ``adet.geometry.BezierCurve``).
  Each endpoint is the mean-line point (inlet: on axis z=0 at radius
  ``R_MEAN_IN``; outlet: at z=``AXIAL_LENGTH``, radius ``R_MEAN_OUT``)
  offset by +-H/2 along the local normal to the prescribed meridional
  angle there, so the inlet/outlet channel height is always measured
  perpendicular to the flow direction, not hard-coded as purely radial
  or purely axial -- this is what lets ``MERID_ANGLE_IN``/``_OUT`` alone
  reshape the channel consistently (it reduces to the "same z at inlet,
  same R at outlet" picture only in the special case
  ``MERID_ANGLE_IN = 0``, ``MERID_ANGLE_OUT = 90 deg``).
"""

import CoolProp as cp
import numpy as np
from matplotlib import pyplot as plt

from adet.assemblers import CasadiSystem
from adet.equations.base_equation import EquationBase, EquationConfig
from adet.fluid.ideal_eos import IdealGasState
from adet.geometry import BezierCurve
from adet.solution import solve_root_problem
from adet.variables import NodeVariables, VariableEnum
from adet.varspec import VarSpec

# ============================================================
# 1. User settings and problem data
# ============================================================
R_MEAN_IN = 0.05  # [m] mean-line radius at the channel inlet
R_MEAN_OUT = 0.130  # [m] mean-line radius at the channel outlet

# Channel (blade) height, measured perpendicular to the local meridional
# direction -- not necessarily radial at inlet or axial at outlet; see
# the note at the top of this file and how hub/shroud are built below.
H_IN = 0.026  # [m] inlet channel height
H_OUT = 0.012  # [m] exit channel height

# Meridional (tangent) angle, shared by hub and shroud (BezierCurve
# convention: 0 deg = axial +z, 90 deg = radial outward).
MERID_ANGLE_IN = np.radians(0.0)  # [rad]
MERID_ANGLE_OUT = np.radians(90.0)  # [rad]

AXIAL_LENGTH = 0.08  # [m] axial length of the mean line (z=0 to z=AXIAL_LENGTH)

N_STREAM = 21  # number of streamwise stations (inlet to outlet)
N_SPAN = 9  # number of spanwise stations (hub to shroud)

RPM = 10_000.0  # [rev/min]
OMEGA = RPM * 2 * np.pi / 60  # [rad/s]

MASS_FLOW = 1.0  # [kg/s]
T0_IN = 293.15  # [K] inlet total temperature
P0_IN = 101_325.0  # [Pa] inlet total pressure

# Ideal-gas air.
GAMMA_AIR = 1.4
R_AIR = 287.05  # [J / (kg K)]
VISCOSITY_AIR = 1.8e-5  # [Pa s], unused here but required by IdealGasState

# Arbitrary prescribed flow-angle field beta(s, eta) (not solved for --
# see module docstring). The hub/shroud beta is given at the inlet and
# at the outlet and linearly distributed across the span; at each
# spanwise station, those two (hub-to-shroud-interpolated) local values
# are then used as the inlet/outlet endpoints of a streamwise parabola
# (with an adjustable, spanwise-uniform mid-passage bulge).
BETA_HUB_IN = np.radians(-10.0)  # [rad] beta at the hub, inlet
BETA_SHROUD_IN = np.radians(-50.0)  # [rad] beta at the shroud, inlet
BETA_HUB_OUT = np.radians(-14.0)  # [rad] beta at the hub, outlet
BETA_SHROUD_OUT = np.radians(-14.0)  # [rad] beta at the shroud, outlet
BETA_BULGE = np.radians(-25.0)  # [rad] extra mid-passage deviation

# Hub and shroud are the mean line offset by +-H/2 along its local
# normal direction (rotate the meridional tangent (cos th, sin th) by
# +90 deg: (-sin th, cos th)). This makes the offset purely radial when
# th=0 (axial inlet) and purely axial when th=90 deg (radial outlet) --
# recovering the usual picture at those limits -- while still rotating
# consistently with MERID_ANGLE_IN/OUT at any other angle, which a fixed
# radial/axial offset would not.
Z_HUB_IN = 0.5 * H_IN * np.sin(MERID_ANGLE_IN)
R_HUB_IN = R_MEAN_IN - 0.5 * H_IN * np.cos(MERID_ANGLE_IN)
Z_SHROUD_IN = -0.5 * H_IN * np.sin(MERID_ANGLE_IN)
R_SHROUD_IN = R_MEAN_IN + 0.5 * H_IN * np.cos(MERID_ANGLE_IN)

Z_HUB_OUT = AXIAL_LENGTH + 0.5 * H_OUT * np.sin(MERID_ANGLE_OUT)
R_HUB_OUT = R_MEAN_OUT - 0.5 * H_OUT * np.cos(MERID_ANGLE_OUT)
Z_SHROUD_OUT = AXIAL_LENGTH - 0.5 * H_OUT * np.sin(MERID_ANGLE_OUT)
R_SHROUD_OUT = R_MEAN_OUT + 0.5 * H_OUT * np.cos(MERID_ANGLE_OUT)

# ============================================================
# 2. Custom classes and functions
# ============================================================


def resample_by_arc_length(
    curve: BezierCurve, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a (finely discretized) curve at ``n_points`` equally
    spaced arc-length stations."""
    ds = np.hypot(np.diff(curve.z_coords), np.diff(curve.r_coords))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_stations = np.linspace(0.0, s[-1], n_points)
    z = np.interp(s_stations, s, curve.z_coords)
    r = np.interp(s_stations, s, curve.r_coords)
    return z, r


def build_meridional_grid(
    hub_z: np.ndarray,
    hub_r: np.ndarray,
    shroud_z: np.ndarray,
    shroud_r: np.ndarray,
    n_span: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid of intermediate quasi streamsurfaces (Figure 3.5), obtained
    by linear interpolation along the hub-to-shroud quasi-orthogonals.

    Returns
    -------
    Z, R : ndarray, shape (n_stream, n_span)
        Meridional-plane coordinates; axis 0 is streamwise (hub/shroud
        curve direction), axis 1 is spanwise (hub -> shroud).
    """
    eta = np.linspace(0.0, 1.0, n_span)[None, :]
    z = hub_z[:, None] * (1.0 - eta) + shroud_z[:, None] * eta
    r = hub_r[:, None] * (1.0 - eta) + shroud_r[:, None] * eta
    return z, r


def spanwise_normal_curvature(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Normal curvature :math:`1/\\mathfrak{R}_n` of each quasi
    streamsurface (fixed spanwise index), projected onto the local
    hub-to-shroud direction -- the signed quantity entering Eq. (3.3).

    Parameters
    ----------
    z, r : ndarray, shape (n_stream, n_span)
        Meridional grid coordinates.

    Returns
    -------
    inv_Rn : ndarray, shape (n_stream, n_span)
    """
    n_stream, n_span = z.shape

    # Local hub -> shroud unit direction at each streamwise station.
    n_z = z[:, -1] - z[:, 0]
    n_r = r[:, -1] - r[:, 0]
    n_mag = np.hypot(n_z, n_r)
    n_z, n_r = n_z / n_mag, n_r / n_mag

    inv_Rn = np.empty((n_stream, n_span))
    for j in range(n_span):
        ds = np.hypot(np.diff(z[:, j]), np.diff(r[:, j]))
        s = np.concatenate([[0.0], np.cumsum(ds)])
        dz_ds = np.gradient(z[:, j], s)
        dr_ds = np.gradient(r[:, j], s)
        d2z_ds2 = np.gradient(dz_ds, s)
        d2r_ds2 = np.gradient(dr_ds, s)
        inv_Rn[:, j] = d2z_ds2 * n_z + d2r_ds2 * n_r

    return inv_Rn


def midline_tangent_angle(z_mid: np.ndarray, r_mid: np.ndarray) -> np.ndarray:
    """Local meridional streamline tangent angle :math:`\\gamma(s)`
    (0 = axial, 90 deg = radial), evaluated on the midspan track and
    used -- spanwise-uniform -- as the :math:`\\cos\\gamma` projection
    in Eq. (3.1)/(3.3)."""
    ds = np.hypot(np.diff(z_mid), np.diff(r_mid))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    dz_ds = np.gradient(z_mid, s)
    dr_ds = np.gradient(r_mid, s)
    return np.arctan2(dr_ds, dz_ds)


def parabolic_beta_field(
    s_hat: np.ndarray, beta_in: np.ndarray, beta_out: np.ndarray
) -> np.ndarray:
    """Flow-angle field, streamwise a parabola (with an adjustable,
    spanwise-uniform mid-passage bulge) between the given per-span
    inlet/outlet beta values.

    Parameters
    ----------
    s_hat : ndarray, shape (n_stream,)
        Streamwise fraction (0 at inlet, 1 at outlet).
    beta_in, beta_out : ndarray, shape (n_span,)
        Local (per-span) inlet/outlet flow angle.

    Returns
    -------
    beta : ndarray, shape (n_stream, n_span)
    """
    s = s_hat[:, None]
    return (
        beta_in[None, :]
        + (beta_out - beta_in)[None, :] * s
        + BETA_BULGE * s * (1.0 - s)
    )


n0 = NodeVariables(0)
n1 = NodeVariables(1)


class MeridionalFlowVariables(VariableEnum):
    """Custom variable for the Eq. (3.6) mass-flow closure, not part of
    ADeT's built-in variable library."""

    CumFlow = VarSpec('cum_mdot', 'kg / s', 0.5)
    """Mass flow accumulated from the hub up to this spanwise station."""


mf0 = MeridionalFlowVariables(0)
mf1 = MeridionalFlowVariables(1)


class LocalThermoState(EquationBase):
    """Static density from rothalpy conservation (Van den Braembussche
    Eq. 1.68) and the ideal-gas EOS at constant (inlet) entropy:

    .. math::
        h = I_0 + \\frac{U^2}{2} - \\frac{\\widetilde{W}^2}{2}, \\qquad
        \\rho = \\rho_{EOS}(h, s_{in})
    """

    config = EquationConfig(manual_units=('kg / m**3',))

    def __init__(
        self,
        blade_speed: float,
        rothalpy: float,
        entropy: float,
        gas_state: IdealGasState,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.u = blade_speed
        self.rothalpy = rothalpy
        self.entropy = entropy
        self.gas_state = gas_state

    def residual(self, w0: n0.kin.W_mag.Hint, rho0: n0.stc.Density.Hint):
        h_static = self.rothalpy + self.u**2 / 2 - w0**2 / 2
        self.gas_state.update(cp.HmassSmass_INPUTS, h_static, self.entropy)
        return rho0 - self.gas_state.rhomass()


class MeridionalMomentumMarch(EquationBase):
    """Trapezoidal finite-difference form of the hub-to-shroud force
    balance between two neighboring spanwise grid points, Van den
    Braembussche Eq. (3.3) with the Coriolis term's sign corrected
    against Katsanis (1964) -- see the module docstring."""

    config = EquationConfig(manual_units=('m / s',))

    def __init__(
        self,
        delta_n: float,
        r0: float,
        r1: float,
        beta0: float,
        beta1: float,
        gamma: float,
        omega: float,
        inv_rn0: float,
        inv_rn1: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.delta_n = delta_n
        self.r0, self.r1 = r0, r1
        self.beta0, self.beta1 = beta0, beta1
        self.gamma = gamma
        self.omega = omega
        self.inv_rn0, self.inv_rn1 = inv_rn0, inv_rn1

    def _rhs(self, w, beta: float, inv_rn: float, r: float):
        return w * np.cos(beta) ** 2 * inv_rn - np.cos(self.gamma) * np.sin(beta) * (
            2 * self.omega + w * np.sin(beta) / r
        )

    def residual(self, w0: n0.kin.W_mag.Hint, w1: n1.kin.W_mag.Hint):
        rhs0 = self._rhs(w0, self.beta0, self.inv_rn0, self.r0)
        rhs1 = self._rhs(w1, self.beta1, self.inv_rn1, self.r1)
        return (w1 - w0) - self.delta_n * 0.5 * (rhs0 + rhs1)


class MassFluxAccumulation(EquationBase):
    """Trapezoidal, chained form of Eq. (3.6): cumulative meridional
    mass flow from the hub up to this spanwise station, between two
    neighboring spanwise grid points."""

    config = EquationConfig(manual_units=('kg / s',))

    def __init__(
        self, delta_n: float, r0: float, r1: float, beta0: float, beta1: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.delta_n = delta_n
        self.r0, self.r1 = r0, r1
        self.beta0, self.beta1 = beta0, beta1

    def residual(
        self,
        w0: n0.kin.W_mag.Hint,
        rho0: n0.stc.Density.Hint,
        cf0: mf0.CumFlow.Hint,
        w1: n1.kin.W_mag.Hint,
        rho1: n1.stc.Density.Hint,
        cf1: mf1.CumFlow.Hint,
    ):
        flux0 = rho0 * w0 * np.cos(self.beta0) * self.r0
        flux1 = rho1 * w1 * np.cos(self.beta1) * self.r1
        return cf1 - cf0 - 2 * np.pi * 0.5 * (flux0 + flux1) * self.delta_n


# ============================================================
# 3. System building and solution
# ============================================================
hub_curve = BezierCurve(
    z_in=Z_HUB_IN,
    z_out=Z_HUB_OUT,
    radius_in=R_HUB_IN,
    radius_out=R_HUB_OUT,
    angle_in=MERID_ANGLE_IN,
    angle_out=MERID_ANGLE_OUT,
    n_points=400,
)
shroud_curve = BezierCurve(
    z_in=Z_SHROUD_IN,
    z_out=Z_SHROUD_OUT,
    radius_in=R_SHROUD_IN,
    radius_out=R_SHROUD_OUT,
    angle_in=MERID_ANGLE_IN,
    angle_out=MERID_ANGLE_OUT,
    n_points=400,
)

hub_z, hub_r = resample_by_arc_length(hub_curve, N_STREAM)
shroud_z, shroud_r = resample_by_arc_length(shroud_curve, N_STREAM)

Z, R = build_meridional_grid(hub_z, hub_r, shroud_z, shroud_r, N_SPAN)
INV_RN = spanwise_normal_curvature(Z, R)
DELTA_N = np.hypot(np.diff(Z, axis=1), np.diff(R, axis=1))

mid_z, mid_r = 0.5 * (hub_z + shroud_z), 0.5 * (hub_r + shroud_r)
GAMMA = midline_tangent_angle(mid_z, mid_r)
MID_S = np.concatenate(
    [[0.0], np.cumsum(np.hypot(np.diff(mid_z), np.diff(mid_r)))]
)  # meridional (midline) arc length [m], for plotting
S_HAT = np.linspace(0.0, 1.0, N_STREAM)  # hub/shroud arc-length fraction
ETA = np.linspace(0.0, 1.0, N_SPAN)  # spanwise fraction (hub -> shroud)

BETA_IN_SPAN = BETA_HUB_IN + (BETA_SHROUD_IN - BETA_HUB_IN) * ETA
BETA_OUT_SPAN = BETA_HUB_OUT + (BETA_SHROUD_OUT - BETA_HUB_OUT) * ETA
BETA = parabolic_beta_field(S_HAT, BETA_IN_SPAN, BETA_OUT_SPAN)  # (N_STREAM, N_SPAN)

U = OMEGA * R  # blade speed [m/s], purely geometric (no feedback)

# Shared IdealGasState instance: its symbolic (h, s) -> rho solution is
# built (and sympy-cached) once and reused by every node below.
GAS_STATE = IdealGasState(gamma=GAMMA_AIR, gas_constant=R_AIR, viscosity=VISCOSITY_AIR)

# Inlet total state (ideal-gas air): rothalpy and entropy are uniform,
# global constants (see module docstring), with no inlet prewhirl so
# that the rothalpy I0 equals the inlet total enthalpy h0.
GAS_STATE.update(cp.PT_INPUTS, P0_IN, T0_IN)
ROTHALPY = GAS_STATE.hmass()  # [J/kg]
ENTROPY_IN = GAS_STATE.smass()  # [J/(kg K)]


def node_index(i: int, j: int) -> int:
    return i * N_SPAN + j


N_NODES = N_STREAM * N_SPAN
nodes = [NodeVariables(k) for k in range(N_NODES)]
mf_nodes = [MeridionalFlowVariables(k) for k in range(N_NODES)]

system = CasadiSystem(num_span=1)

EQUATIONS = {}
for i in range(N_STREAM):
    gamma_i = GAMMA[i]

    for j in range(N_SPAN):
        k = node_index(i, j)
        EQUATIONS[
            LocalThermoState(
                blade_speed=U[i, j],
                rothalpy=ROTHALPY,
                entropy=ENTROPY_IN,
                gas_state=GAS_STATE,
            )
        ] = k

    for j in range(1, N_SPAN):
        k0, k1 = node_index(i, j - 1), node_index(i, j)
        EQUATIONS[
            MeridionalMomentumMarch(
                delta_n=DELTA_N[i, j - 1],
                r0=R[i, j - 1],
                r1=R[i, j],
                beta0=BETA[i, j - 1],
                beta1=BETA[i, j],
                gamma=gamma_i,
                omega=OMEGA,
                inv_rn0=INV_RN[i, j - 1],
                inv_rn1=INV_RN[i, j],
            )
        ] = (k0, k1)
        EQUATIONS[
            MassFluxAccumulation(
                delta_n=DELTA_N[i, j - 1],
                r0=R[i, j - 1],
                r1=R[i, j],
                beta0=BETA[i, j - 1],
                beta1=BETA[i, j],
            )
        ] = (k0, k1)

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

BOUNDARY_CONDITIONS = {}
for i in range(N_STREAM):
    BOUNDARY_CONDITIONS[mf_nodes[node_index(i, 0)].CumFlow] = 0.0
    BOUNDARY_CONDITIONS[mf_nodes[node_index(i, N_SPAN - 1)].CumFlow] = MASS_FLOW

system.add_boundary_conditions(BOUNDARY_CONDITIONS)

system.build()

# Guesses: a flow-based velocity scale, a representative air density,
# and a linear hub-to-shroud ramp for the cumulative mass flow.
area_in_est = 2 * np.pi * R_MEAN_IN * H_IN
w_guess = MASS_FLOW / (1.2 * area_in_est)

manual_guess = {}
for i in range(N_STREAM):
    for j in range(N_SPAN):
        k = node_index(i, j)
        manual_guess[nodes[k].kin.W_mag] = w_guess
        manual_guess[nodes[k].stc.Density] = 1.2
        manual_guess[mf_nodes[k].CumFlow] = MASS_FLOW * j / (N_SPAN - 1)

x0 = system.get_guess(manual_guess, fallback=w_guess)
kn = system.get_boundary_conds()
bnd = system.get_bounds()

rootfinder = system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False, 'ipopt.tol': 1e-8, 'ipopt.max_iter': 1000},
)
solution = solve_root_problem(rootfinder, x0, kn, bnd, suppress_output=False)

rootfinder_kinsol = system.make_rootfinder('kinsol')
solution = solve_root_problem(rootfinder_kinsol, solution, kn)

sol_dict = system.sol_to_dict(solution)

W_grid = np.array(
    [
        [sol_dict[nodes[node_index(i, j)].kin.W_mag][0] for j in range(N_SPAN)]
        for i in range(N_STREAM)
    ]
)
RHO_grid = np.array(
    [
        [sol_dict[nodes[node_index(i, j)].stc.Density][0] for j in range(N_SPAN)]
        for i in range(N_STREAM)
    ]
)
Wm_grid = W_grid * np.cos(BETA)
Wt_grid = W_grid * np.sin(BETA)  # tangential relative velocity component

# Static and relative-total temperature/pressure, from the same (h, s)
# closure as LocalThermoState (rothalpy conservation + constant inlet
# entropy): h0,rel = I0 + U^2/2 is independent of W (it's just the
# rothalpy definition rearranged), so it's recomputed directly rather
# than re-deriving it from h_static + W^2/2.
H_STATIC_grid = ROTHALPY + U**2 / 2 - W_grid**2 / 2
GAS_STATE.update(cp.HmassSmass_INPUTS, H_STATIC_grid, ENTROPY_IN)
T_grid = GAS_STATE.T()
P_grid = GAS_STATE.p()

H0_REL_grid = ROTHALPY + U**2 / 2
GAS_STATE.update(cp.HmassSmass_INPUTS, H0_REL_grid, ENTROPY_IN)
T0_grid = GAS_STATE.T()
P0_grid = GAS_STATE.p()

# Absolute-total temperature/pressure: same static enthalpy, but
# stagnating the absolute velocity V = Vm + Vu (Vm = Wm, Vu = Wu + U)
# instead of the relative one.
Vu_grid = Wt_grid + U
H0_ABS_grid = H_STATIC_grid + (Wm_grid**2 + Vu_grid**2) / 2
GAS_STATE.update(cp.HmassSmass_INPUTS, H0_ABS_grid, ENTROPY_IN)
T0_ABS_grid = GAS_STATE.T()
P0_ABS_grid = GAS_STATE.p()

print(
    f'Meridional velocity: hub mean = {Wm_grid[:, 0].mean():.2f} m/s, '
    f'shroud mean = {Wm_grid[:, -1].mean():.2f} m/s'
)

# ============================================================
# 4. Diagnostic plots
# ============================================================
if __name__ == '__main__':
    # Per-panel size shared by every figure below, so they all look
    # consistent and none of them overflow the screen.
    PANEL_W, PANEL_H = 5.0, 4.0  # [in]

    def plot_contour_row(
        panels: list[tuple[np.ndarray, str]],
        suptitle: str,
        cbar_label: str,
        shared_scale: bool,
    ):
        """One row of meridional-plane contourf panels (hub/shroud
        contours and quasi-orthogonals overlaid). If ``shared_scale``,
        every panel uses the same color scale (a single colormap,
        directly comparable across panels) and one shared colorbar;
        otherwise each panel gets its own independent scale/colorbar.
        """
        n = len(panels)
        fig, axs = plt.subplots(
            1, n, figsize=(n * PANEL_W, PANEL_H), sharex=True, sharey=True
        )
        axs = np.atleast_1d(axs)

        levels = 30
        if shared_scale:
            vmin = min(field.min() for field, _ in panels)
            vmax = max(field.max() for field, _ in panels)
            levels = np.linspace(vmin, vmax, 31)

        contour = None
        for ax, (field, title) in zip(axs, panels):
            contour = ax.contourf(Z, R, field, levels=levels, cmap='viridis')
            ax.plot(hub_z, hub_r, color='k', linewidth=1.5, label='Hub')
            ax.plot(
                shroud_z,
                shroud_r,
                color='k',
                linewidth=1.5,
                linestyle='--',
                label='Shroud',
            )
            for i in range(0, N_STREAM, max(1, N_STREAM // 15)):
                ax.plot(Z[i, :], R[i, :], color='white', linewidth=0.4)

            ax.set_xlabel('z [m]')
            ax.set_ylabel('R [m]')
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.legend(loc='lower right')
            if not shared_scale:
                fig.colorbar(contour, ax=ax, label=cbar_label)

        fig.suptitle(suptitle)
        if shared_scale:
            assert contour is not None
            fig.colorbar(contour, ax=axs.tolist(), label=cbar_label)
        else:
            fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.92))

        return fig, axs

    # --- Figure 1: velocity fields on the meridional plane ---
    fig1, axs1 = plot_contour_row(
        [
            (Wm_grid, 'Meridional velocity $W_m$'),
            (Wt_grid, 'Tangential relative velocity $W_u$'),
            (W_grid, r'Overall relative velocity $\widetilde{W}$'),
        ],
        suptitle='Hub-to-shroud velocity distribution (full Eq. 3.3, compressible)',
        cbar_label='Velocity [m/s]',
        shared_scale=False,
    )

    # --- Figure 2: the three RHS terms of Eq. (3.3), plus their sum ---
    # dW/dn = [curvature] - [Coriolis] - [swirl], see the module docstring
    # (Coriolis sign corrected against Katsanis 1964).
    CURV_grid = W_grid * np.cos(BETA) ** 2 * INV_RN
    CORIOLIS_grid = -np.cos(GAMMA)[:, None] * np.sin(BETA) * 2 * OMEGA
    SWIRL_grid = -np.cos(GAMMA)[:, None] * np.sin(BETA) ** 2 * W_grid / R
    NET_grid = CURV_grid + CORIOLIS_grid + SWIRL_grid  # = dW/dn

    fig_terms, axs_terms = plot_contour_row(
        [
            (CURV_grid, r'Curvature: $\widetilde{W}\cos^2\!\beta/\mathfrak{R}_n$'),
            (CORIOLIS_grid, r'Coriolis: $-2\Omega\cos\gamma\sin\beta$'),
            (SWIRL_grid, r'Swirl: $-\cos\gamma\sin^2\!\beta\,\widetilde{W}/R$'),
            (NET_grid, r'Sum: $\partial \widetilde{W}/\partial n$'),
        ],
        suptitle='Eq. (3.3) term breakdown (Coriolis sign per Katsanis 1964)',
        cbar_label=r'[1/s]',
        shared_scale=False,
    )

    # --- Figure 3: computational grid and the prescribed beta(s) law ---
    fig2, axs2 = plt.subplots(1, 2, figsize=(2 * PANEL_W, PANEL_H))

    ax_grid = axs2[0]
    for i in range(N_STREAM):  # quasi-orthogonals
        ax_grid.plot(Z[i, :], R[i, :], color='tab:blue', linewidth=0.5)
    for j in range(N_SPAN):  # quasi streamsurfaces
        ax_grid.plot(Z[:, j], R[:, j], color='tab:orange', linewidth=0.5)
    ax_grid.plot(hub_z, hub_r, color='k', linewidth=1.5, label='Hub')
    ax_grid.plot(
        shroud_z, shroud_r, color='k', linewidth=1.5, linestyle='--', label='Shroud'
    )
    ax_grid.set_xlabel('z [m]')
    ax_grid.set_ylabel('R [m]')
    ax_grid.set_aspect('equal')
    ax_grid.set_title(f'Meridional grid ({N_STREAM} x {N_SPAN})')
    ax_grid.legend(loc='lower right')

    ax_beta = axs2[1]
    for j, label in ((0, 'Hub'), (N_SPAN // 2, 'Mid'), (N_SPAN - 1, 'Shroud')):
        ax_beta.plot(MID_S, np.degrees(BETA[:, j]), '-o', label=label)
    ax_beta.set_xlabel('Meridional distance $s$ [m]')
    ax_beta.set_ylabel(r'Flow angle $\beta$ [deg]')
    ax_beta.set_title(r'Prescribed $\beta(s,\eta)$ field (Eq. 3.3 input, not solved)')
    ax_beta.legend()
    ax_beta.grid(True, alpha=0.3)

    fig2.suptitle('Computational grid and prescribed flow-angle law')
    fig2.tight_layout()

    # --- Figure 4: inlet velocity triangles (hub, mid, shroud) ---
    def plot_velocity_triangle(ax, wm: float, wu: float, u: float, title: str):
        """Draw the solved inlet velocity triangle in the (meridional,
        tangential) plane: O -> W_tip is W, W_tip -> V_tip is U
        (Vu = Wu + U), O -> V_tip is V."""
        origin = np.array([0.0, 0.0])
        w_tip = np.array([wm, wu])
        v_tip = w_tip + np.array([0.0, u])

        for tip, color, label in (
            (w_tip, 'tab:blue', r'$\widetilde{W}$'),
            (v_tip, 'tab:green', '$V$'),
        ):
            ax.annotate(
                '',
                xy=tip,
                xytext=origin,
                arrowprops={'arrowstyle': '-|>', 'color': color, 'linewidth': 2},
            )
            ax.text(*(0.5 * tip), label, color=color, fontsize=11)

        ax.annotate(
            '',
            xy=v_tip,
            xytext=w_tip,
            arrowprops={'arrowstyle': '-|>', 'color': 'tab:red', 'linewidth': 2},
        )
        ax.text(*(w_tip + 0.5 * (v_tip - w_tip)), '$U$', color='tab:red', fontsize=11)

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Meridional [m/s]')
        ax.set_ylabel('Tangential [m/s]')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    span_stations = ((0, 'Hub'), (N_SPAN // 2, 'Mid'), (N_SPAN - 1, 'Shroud'))
    j_stations = [j for j, _ in span_stations]
    wm_in = Wm_grid[0, j_stations]
    wu_in = Wt_grid[0, j_stations]
    u_in = U[0, j_stations]
    vu_in = wu_in + u_in

    x_max = wm_in.max()
    y_min, y_max = wu_in.min(), max(u_in.max(), vu_in.max(), 0.0)
    x_lim = (-0.15 * x_max, 1.15 * x_max)
    y_lim = (y_min - 0.15 * (y_max - y_min), y_max + 0.15 * (y_max - y_min))

    fig3, axs3 = plt.subplots(1, 3, figsize=(3 * PANEL_W, PANEL_H))
    for ax, (j, label) in zip(axs3, span_stations):
        plot_velocity_triangle(
            ax, Wm_grid[0, j], Wt_grid[0, j], U[0, j], f'{label} ($\\eta$={ETA[j]:.2f})'
        )
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

    fig3.suptitle(r'Inlet velocity triangles (solved $\widetilde{W}$, $\beta$)')
    fig3.tight_layout()

    # --- Figure 5: temperature (static, relative-total, absolute-total) ---
    fig4, axs4 = plot_contour_row(
        [
            (T_grid, 'Static'),
            (T0_grid, 'Relative-total'),
            (T0_ABS_grid, 'Absolute-total'),
        ],
        suptitle=(
            'Temperature fields (ideal-gas EOS, rothalpy conservation'
            ' + constant inlet entropy)'
        ),
        cbar_label='Temperature $T$ [K]',
        shared_scale=True,
    )

    # --- Figure 6: pressure (static, relative-total, absolute-total) ---
    fig5, axs5 = plot_contour_row(
        [
            (P_grid / 1e3, 'Static'),
            (P0_grid / 1e3, 'Relative-total'),
            (P0_ABS_grid / 1e3, 'Absolute-total'),
        ],
        suptitle=(
            'Pressure fields (ideal-gas EOS, rothalpy conservation'
            ' + constant inlet entropy)'
        ),
        cbar_label='Pressure $p$ [kPa]',
        shared_scale=True,
    )

    plt.show()
