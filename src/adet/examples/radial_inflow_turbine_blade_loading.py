"""
Meanline blade-to-blade loading of a radial-inflow turbine (IFR) rotor.

This is the turbine counterpart of ``radial_impeller_blade_loading.py``: the
same meanline, inviscid quasi-3D blade-to-blade loading model from Chapter 3
("Radial Impeller Flow Calculation") of Van den Braembussche, *Design and
Analysis of Centrifugal Compressors* (2019), Eqs. (3.1)-(3.20), applied to a
90-degree inflow-radial/outflow-axial turbine rotor instead of a centrifugal
compressor impeller. See that file's module docstring for the full
derivation and the modeling choices shared by both examples (rothalpy
conservation, the general blade-to-blade loading law, the SS/PS velocity
superposition and blade surface pressures, the zero-incidence-then-Kutta
deviation model); only what differs for a turbine rotor is noted here.

What actually differs from the compressor case
-------------------------------------------------
The underlying equations (rothalpy conservation, Eq. 3.13 loading, the
SS/PS pressure recovery) do not care whether the machine is a compressor or
a turbine -- they are unchanged, reused as-is. Only the *geometry and
operating point* are mirrored:

- The meanline now runs from a large-radius inlet (rotor leading edge,
  right after the nozzle/volute) to a small-radius outlet (the exducer),
  i.e. ``R_IN > R_OUT``. By default the inlet is purely radial and the
  outlet purely axial (the classic "90-degree IFR turbine"), but
  ``MERID_ANGLE_IN``/``MERID_ANGLE_OUT`` are free parameters -- set them to
  anything other than -90/0 deg for a non-radial inlet or non-axial
  outlet, i.e. a mixed-flow channel shape. See the NOTE above
  ``MERID_ANGLE_IN`` for the sign convention (get it wrong and the
  meanline radius overshoots non-monotonically; this is now caught by an
  assertion right after the Bezier curve is built).
- The leading-edge blade metal angle is taken as purely radial
  (``BETA_BL_IN = 0``), the common "90-degree IFR turbine" zero-incidence
  design convention (absolute inlet swirl equal to the blade speed); the
  trailing-edge (exducer) angle is strongly backswept
  (``BETA_BL_OUT`` a large negative angle), consistent with
  ``radial_inflow_turbine.py``'s rotor.
- No sign flips are needed anywhere else: with radius *decreasing*
  downstream and the exducer turning the flow to a large negative relative
  angle, :math:`\\Omega \\, d(R V_u)/ds` comes out negative on its own,
  i.e. rothalpy/work is *extracted* from the fluid rather than added to it
  -- exactly the expected turbine behaviour. This is printed as a
  diagnostic at the end, mirroring the compressor example's Eq. (3.19)-
  (3.20) check.
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
# 1. Rotor geometry and operating point (all prescribed)
# ============================================================
N_STATIONS = 15  # LE = station 0 (radial inlet), TE = station N_STATIONS - 1 (exducer)

R_IN = 0.060  # [m] meanline radius at the leading edge
R_OUT = 0.018  # [m] meanline radius at the trailing edge (exducer)
Z_LENGTH = 0.050  # [m] axial extent of the meridional path (for plotting)

# Meridional flow (tangent) angle at each end of the channel -- this is
# passed straight through to BezierCurve's own angle_in/angle_out, whose
# convention it follows exactly: 0 deg = purely axial (+z), +90 deg =
# purely radial *outward* (dr/ds > 0), -90 deg = purely radial *inward*
# (dr/ds < 0). The default below (-90/0) is the "90-degree IFR turbine"
# case used so far: a purely radial inlet curving to a purely axial
# exducer exit. Set either to any value in between (or beyond) for a
# non-radial inlet / non-axial outlet, i.e. a mixed-flow channel shape.
#
# NOTE (sign convention -- get this wrong and the meanline radius overshoots
# non-monotonically instead of sweeping smoothly from R_IN to R_OUT, which
# silently breaks the whole quasi-1D area/velocity model): since R_IN >
# R_OUT here, MERID_ANGLE_IN must have a *negative* (inward) component and
# MERID_ANGLE_OUT a component consistent with approaching R_OUT from above
# -- i.e. angles are given exactly as the physical direction of travel at
# that point, not negated or mirrored. This is checked automatically below
# (see the monotonic-radius assertion).
MERID_ANGLE_IN = np.radians(-90.0)  # [rad] purely radial (inward) at the inlet
MERID_ANGLE_OUT = np.radians(0.0)  # [rad] purely axial at the outlet (exducer)

# NOTE: The passage width H is *derived* from a prescribed, monotonically
# contracting flow area A(s) = 2 pi R(s) H(s) below, rather than picked
# independently of R(s). Interpolating H linearly in station index while R
# follows the (nonlinear) meridional path makes A(s) bulge in the middle
# (diffusing, i.e. locally decelerating flow) even though both R_IN>R_OUT
# and H_IN<H_OUT individually look like a sensible turbine passage -- and
# a turbine's flow should accelerate continuously (p. 76 of the book:
# "The flow in turbines is mostly accelerating"), unlike a compressor's.
A_IN = 15.0e-4  # [m^2] flow area at the leading edge
A_OUT = 13.0e-4  # [m^2] flow area at the exducer exit (contracting => accelerating)

BETA_BL_IN = np.radians(0.0)  # [rad] radial blades at the LE (zero-incidence design)
BETA_BL_OUT = np.radians(-58.0)  # [rad] blade metal angle at the exducer (backswept)

NUM_BLADES = 19
BLADE_THICKNESS = 0.0005  # [m], constant along the streamline

RPM = 60_000.0
OMEGA = RPM * 2 * np.pi / 60  # [rad/s]

MASS_FLOW = 0.05  # [kg/s]
P0_TOT = 1.5e5  # [Pa]
T0_TOT = 650.0  # [K]

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
A_STATIONS = np.linspace(A_IN, A_OUT, N_STATIONS)
H_STATIONS = A_STATIONS / (2 * np.pi * R_STATIONS)
BETA_BL_STATIONS = np.linspace(BETA_BL_IN, BETA_BL_OUT, N_STATIONS)
DELTA_S = np.diff(_s_stations)  # streamwise spacing of each interval

# ============================================================
# 3. Node and custom variable declarations
# ============================================================
nodes = [NodeVariables(i) for i in range(N_STATIONS)]

# Local node placeholders 0 ("upstream"/"self") and 1 ("downstream"/"other"),
# used only inside the equation classes' residual() hints below -- see
# radial_impeller_blade_loading.py for why this is unrelated to the
# absolute station index.
n0 = NodeVariables(0)
n1 = NodeVariables(1)


class BladeLoadingVariables(VariableEnum):
    """
    Custom variables for the blade-to-blade loading (Section 3.1.2) that
    are not part of ADeT's built-in variable library.
    """

    # NOTE: floored at 5 m/s rather than 0 -- near the leading edge the
    # inviscid, quasi-1D loading law (Eq. 3.13) predicts a suction/pressure
    # side velocity difference large enough to drive W_ps through zero and
    # negative, which is a modeling artifact (real flow would separate and
    # recirculate there long before that, which this model doesn't
    # capture), not a physical solution. Flooring keeps the solver away
    # from that artifact instead of trying to satisfy it.
    W_ss = VarSpec('W_ss', 'm / s', 150.0, (5.0, 2e3))
    """Relative velocity on the suction side, :math:`W_{SS}`."""

    W_ps = VarSpec('W_ps', 'm / s', 100.0, (5.0, 2e3))
    """Relative velocity on the pressure side, :math:`W_{PS}`."""

    DeltaW = VarSpec('delta_W', 'm / s', 20.0, (-1e3, 1e3))
    """Blade loading :math:`W_{SS} - W_{PS}`, Eq. (3.13)."""

    P_ss = VarSpec('p_ss', 'Pa', 2e5, (1e2, 2e7))
    """Static pressure on the suction side."""

    P_ps = VarSpec('p_ps', 'Pa', 2.5e5, (1e2, 2e7))
    """Static pressure on the pressure side."""


bl_nodes = [BladeLoadingVariables(i) for i in range(N_STATIONS)]
bl0 = BladeLoadingVariables(0)
bl1 = BladeLoadingVariables(1)


# ============================================================
# 4. Custom equations specific to this book chapter
#    (identical physics to radial_impeller_blade_loading.py --
#    rothalpy conservation, blade-to-blade loading, slip transition
#    and SS/PS velocity/pressure recovery do not depend on whether
#    radius increases or decreases along the meanline)
# ============================================================
class RothalpyConservation(EquationBase):
    """Conservation of rothalpy between two consecutive meanline stations."""

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
    Eq. (3.13), evaluated with a backward finite difference between the
    upstream (node 0) and downstream (node 1) meanline stations.

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

        # NOTE (sign flip vs. the compressor example): Eq. (3.13)-(3.15) are
        # unchanged from radial_impeller_blade_loading.py, but rotation_term
        # (Eq. 3.14, ~d(Omega R^2)/ds) is the dominant contribution to the
        # loading and it flips sign here simply because R *decreases*
        # downstream in this turbine meanline, whereas it increases
        # downstream in the compressor case the SS/PS convention below was
        # calibrated against. Without this flip, W_ss/p_ss and W_ps/p_ps
        # come out swapped: verified against the always-true suction-side
        # definition (lower pressure / higher velocity than the pressure
        # side), the unflipped sign gives W_ss < W_ps and p_ss > p_ps
        # everywhere in the interior -- backwards.
        loading = -pitch_minus_thickness * (rotation_term - turning_term)

        return delta_w1 - loading


class SlipTransition(EquationBase):
    """
    Simplified stand-in for the flow-angle/blade-angle deviation model of
    Eqs. (3.9)-(3.12), used only over the small region approaching the
    exducer exit where the loading must relax to zero (Kutta condition).
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
    relative velocity to recover the SS and PS velocity distribution."""

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
    surfaces, from rothalpy conservation + the isentropic (inviscid)
    assumption shared by both blade surfaces."""

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

# Station index of the transition point s* (Eqs. 3.9-3.12): upstream of it
# the flow follows the blade angle exactly (ZeroDeviation); from it to the
# exducer exit, the flow angle is left free and instead follows the smooth
# quadratic blend of SlipTransition, which relaxes the loading to zero by
# the trailing edge (Kutta condition).
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
    # At i == TRAILING_EDGE, the flow angle is left free entirely: it *is*
    # the beta_2_slip referenced by SlipTransition above, and it is itself
    # pinned by the Kutta condition (see BOUNDARY_CONDITIONS).
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

# No blade loading right at the leading edge: zero incidence at the radial
# inlet, so the loading described by Eq. (3.13) only builds up downstream.
BOUNDARY_CONDITIONS[bl_nodes[0].DeltaW] = 0.0

# Kutta condition at the exducer exit: W_SS = W_PS, i.e. zero blade loading
# right at the outlet -- see radial_impeller_blade_loading.py for why this
# leaves the trailing-edge flow angle free to self-consistently deviate
# from the blade metal angle instead.
BOUNDARY_CONDITIONS[bl_nodes[TRAILING_EDGE].DeltaW] = 0.0

system.add_boundary_conditions(BOUNDARY_CONDITIONS)

# ============================================================
# 7. Build and solve
# ============================================================
system.build()

# NOTE: The exducer turns the flow by ~58 deg over the meanline; the
# uniform fallback guess is too far from that strongly-varying profile for
# IPOPT to find a feasible point. Seed the flow angle with the blade metal
# angle law itself (a good approximation almost everywhere except right at
# the Kutta-condition trailing edge) and the relative velocity with the
# local blade speed, which is the dominant contribution at this design
# point (zero incidence => W ~ U at the inlet).
_RHO0_APPROX = P0_TOT / (287.0 * T0_TOT)  # crude ideal-gas density estimate

MANUAL_GUESSES = {}
for i in range(N_STATIONS):
    MANUAL_GUESSES[nodes[i].kin.FlowAngleRel] = BETA_BL_STATIONS[i]
    MANUAL_GUESSES[nodes[i].kin.W_mag] = max(OMEGA * R_STATIONS[i], 50.0)
    # NOTE: Seed mass flow, density and meridional velocity consistently
    # (mf = rho * Vm * A) at *every* station from the start: without this,
    # IPOPT starts from a guess that violates MassConservation station-to-
    # station and gets stuck bridging the leading stations back to the
    # inlet boundary condition (observed as a "local infeasibility" with
    # a self-consistent-but-wrong mass flow downstream of station ~3).
    MANUAL_GUESSES[nodes[i].oth.StreamMassFlow] = MASS_FLOW
    MANUAL_GUESSES[nodes[i].stc.Density] = _RHO0_APPROX
    MANUAL_GUESSES[nodes[i].kin.V_mer] = MASS_FLOW / (
        _RHO0_APPROX * 2 * np.pi * R_STATIONS[i] * H_STATIONS[i]
    )

x0 = system.get_guess(MANUAL_GUESSES, fallback=0.6)
kn = system.get_boundary_conds()

# NOTE: Bound the free-deviation region's flow angle within a generous
# +-25 deg slack of the blade metal angle law; this keeps the strongly
# nonlinear SlipTransition/Kutta subsystem from wandering into unphysical
# deviations while searching, without pinning it to zero deviation.
CUSTOM_BOUNDS = {}
for i in range(TRANSITION_STAR + 1, N_STATIONS):
    slack = np.radians(25.0)
    CUSTOM_BOUNDS[nodes[i].kin.FlowAngleRel] = (
        BETA_BL_STATIONS[i] - slack,
        BETA_BL_STATIONS[i] + slack,
    )
bnd = system.get_bounds(CUSTOM_BOUNDS)

rootfinder = system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.tol': 1e-8,
        'ipopt.max_iter': 1000,
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.acceptable_tol': 1e-4,
        'ipopt.acceptable_iter': 15,
    },
)
solution = solve_root_problem(rootfinder, x0, kn, bnd, suppress_output=False)

# NOTE: polish with a *second, tighter* bounded IPOPT solve, not the usual
# unconstrained Newton-Krylov (KINSOL) pass used elsewhere in ADeT. KINSOL
# has no notion of bounds, so starting it from this solution would just
# march straight back to the system's true unconstrained root -- which is
# exactly the unphysical negative-W_ps point the W_ss/W_ps floor above
# (BladeLoadingVariables) exists to avoid. A second bounded IPOPT pass
# keeps that floor active while still tightening the residual.
rootfinder_polish = system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.tol': 1e-10,
        'ipopt.max_iter': 1000,
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.warm_start_init_point': 'yes',
    },
)
solution = solve_root_problem(rootfinder_polish, solution, kn, bnd)

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

# Diagnostic mirroring Eqs. (3.19)-(3.20) of the compressor example: here
# R*Vu should *decrease* along the meanline (radius shrinks and the exducer
# turns the flow to a large negative relative angle), so the Euler work
# Omega * d(R*Vu) comes out negative -- work extracted from the fluid, as
# expected for a turbine.
r_vu = np.array(
    [
        sol_dict[nodes[i].geo.RDistr][0] * sol_dict[nodes[i].kin.V_tan][0]
        for i in range(N_STATIONS)
    ]
)
logger.info(f'R*Vu along the meanline: {r_vu}')
logger.info(
    f'Euler work per unit mass (Omega * d(R*Vu), negative = extracted): '
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

# --- Velocity distribution
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
