"""
Meanline blade-to-blade loading of a radial-inflow turbine (IFR) rotor.

Turbine counterpart of ``radial_impeller_blade_loading.py``: same meanline,
inviscid quasi-3D blade-to-blade loading model (Van den Braembussche,
*Design and Analysis of Centrifugal Compressors*, 2019, Ch. 3, Eqs.
3.1-3.20), applied to a 90-deg inflow-radial/outflow-axial turbine rotor.

Differences from the compressor case:
- Meanline runs from large-radius inlet to small-radius exducer outlet
  (``R_IN > R_OUT``), reversed vs. the compressor case.
- Blade metal angle: purely radial at LE (``BETA_BL_IN = 0``), strongly
  backswept at the exducer (matches ``radial_inflow_turbine.py``).
- Rotor inlet does not assume zero incidence: uses
  ``adet.equations.control_volumes.OptimalIncidenceRadialInflowTurbine``
  (Chen & Baines, 1994) for the flow-slip-driven optimum incidence.
"""

import logging

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.equations.base_equation import EquationBase, EquationConfig
from adet.equations.fundamental import (
    BladeBlockage,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.equations.geometrical import AnnulusAreas
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.geometry import BezierCurve
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.equations.control_volumes import OptimalIncidenceRadialInflowTurbine
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables, VariableEnum
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)
setup_logger(logger)

thrm = ThermoVariables()

# ============================================================
# 1. Parameters / inputs
# ============================================================

NUM_BLADES = 16
BLADE_THICKNESS = 0.001  # [m], constant along the streamline

RPM = 100_000.0
OMEGA = RPM * 2 * np.pi / 60  # [rad/s]

MASS_FLOW = 0.2  # [kg/s]
P0_TOT = 3e5  # [Pa]
T0_TOT = 800.0  # [K]i

R_IN = 0.060  # [m] meanline radius at the leading edge
R_OUT = 0.025  # [m] meanline radius at the trailing edge (exducer)
Z_LENGTH = 0.050  # [m] axial extent of the meridional path (for plotting)
H_IN = 0.006
H_OUT = 0.021
# BezierCurve convention: 0 deg = axial +z, +90 = radial outward, -90 =
# radial inward; angles are the physical direction of travel (not mirrored
# between ends -- wrong sign makes the meanline radius overshoot).
MERID_ANGLE_IN = np.radians(-90.0)  # [rad] purely radial (inward) at the inlet
MERID_ANGLE_OUT = np.radians(0.0)  # [rad] purely axial at the outlet (exducer)

BETA_BL_IN = np.radians(0.0)  # [rad] radial blades at the LE (zero-incidence design)
BETA_BL_OUT = np.radians(-58.0)  # [rad] blade metal angle at the exducer (backswept)

# The channel's *total* area is prescribed (via H_IN/H_OUT) to grow
# monotonically end to end, but num_span=1 solves a single streamline at
# midspan, not a flow-area-weighted average -- feeding it the total
# channel area (as before) implicitly assumes the full mass flow crosses
# the full annulus at that one streamline's local conditions, which can
# diverge from what's actually happening at midspan specifically if the
# hub/shroud move apart at a different rate than the middle of the span
# does. Instead, split the span into N_STREAMTUBES equal-height slices at
# every station and solve using only the midspan slice's own share of the
# height and mass flow -- see the precompute block below and
# H_MIDSPAN_STATIONS/MASS_FLOW_MIDSPAN.
N_STREAMTUBES = 5
MASS_FLOW_MIDSPAN = MASS_FLOW / N_STREAMTUBES  # [kg/s] midspan streamtube's share

# ------------------------------------------------------------
# Toggles: slip-transition blend and Kutta (zero blade-loading) condition,
# independently switchable at the LE and TE (see Section 3 for how each
# one reshapes the equation set -- LE_BLEND_END, LAST_FIXED_BETA_STATION,
# TE_BLEND_START, CarryLoadingToLE).
# ------------------------------------------------------------
USE_SLIP_TRANSITION_LE = True  # blend the LE incidence onto the blade-angle law
USE_SLIP_TRANSITION_TE = True  # blend the blade-angle law onto the free TE value
USE_KUTTA_LE = True  # assume zero blade loading (DeltaW=0) at the LE
USE_KUTTA_TE = True  # assume zero blade loading (DeltaW=0, W_SS=W_PS) at the TE

# Streamwise fraction (of TRAILING_EDGE) spanning each SlipTransition blend
# -- only effective when the corresponding toggle above is on.
TRANSITION_FRAC_LE = 0.25
TRANSITION_FRAC_TE = 0.6

N_STATIONS = 25  # LE = station 0 (radial inlet), TE = station N_STATIONS - 1 (exducer)

# Blade metal angle streamwise law beta_bl(s), toggled between a few
# illustrative parametrizations (not tied to a specific reference, unlike
# e.g. the Sauret-table-based law in T100_RIT_VdB.py) -- all blend exactly
# from BETA_BL_IN (s=0) to BETA_BL_OUT (s=S_MAX), see _blade_angle_law().
BETA_BL_PARAM = 'parabolic'  # 'linear' | 'parabolic' | 'circular' | 'piecewise'

# 'piecewise' only: two linear segments meeting at PIECEWISE_SPLIT_FRAC
# (streamwise fraction of S_MAX), slope of the first segment prescribed
# (PIECEWISE_SLOPE1, rad per unit s/S_MAX), slope of the second derived
# from continuity with BETA_BL_OUT at s=S_MAX.
PIECEWISE_SPLIT_FRAC = 0.35
PIECEWISE_SLOPE1 = np.radians(-25.0)  # [rad per unit s/S_MAX]
PIECEWISE_BLEND_HALF_WIDTH = 0.25  # [frac of S_MAX] smoothing window half-width


def _blade_angle_law(frac: np.ndarray, kind: str) -> np.ndarray:
    """Blade metal angle beta_bl(s) for normalized arc length
    ``frac = s / S_MAX`` in [0, 1]."""
    if kind == 'linear':
        shape = frac
        return BETA_BL_IN + shape * (BETA_BL_OUT - BETA_BL_IN)

    if kind == 'parabolic':
        # Zero slope at the LE (matches the purely-radial, zero-incidence
        # LE design intent), quadratic turning thereafter.
        shape = frac**2
        return BETA_BL_IN + shape * (BETA_BL_OUT - BETA_BL_IN)

    if kind == 'circular':
        # Quarter circle: zero slope at the LE, turning concentrated near
        # the TE (steeper than the parabolic law).
        shape = 1.0 - np.sqrt(np.clip(1.0 - frac**2, 0.0, None))
        return BETA_BL_IN + shape * (BETA_BL_OUT - BETA_BL_IN)

    if kind == 'piecewise':
        beta_split = BETA_BL_IN + PIECEWISE_SLOPE1 * PIECEWISE_SPLIT_FRAC
        slope2 = (BETA_BL_OUT - beta_split) / (1.0 - PIECEWISE_SPLIT_FRAC)

        line1 = BETA_BL_IN + PIECEWISE_SLOPE1 * frac
        line2 = beta_split + slope2 * (frac - PIECEWISE_SPLIT_FRAC)

        # Cubic smoothstep blend (zero slope at both ends of the window)
        # instead of a sharp corner at the split.
        hw = PIECEWISE_BLEND_HALF_WIDTH
        t = np.clip((frac - (PIECEWISE_SPLIT_FRAC - hw)) / (2 * hw), 0.0, 1.0)
        w = t * t * (3 - 2 * t)
        return (1.0 - w) * line1 + w * line2

    raise ValueError(f'Unknown BETA_BL_PARAM: {kind!r}')


class BladeChannelGeometry:
    """Blade span H(s) and the resulting flow area along the meanline,
    from the prescribed LE/TE spans (H_IN, H_OUT) and the meridional
    streamline R(s) -- replaces guessing a flow-area law directly (as this
    file previously did: ``A_STATIONS = linspace(A_IN, A_OUT, ...)``,
    ``H_STATIONS = A_STATIONS / (2*pi*R_STATIONS)``, unrelated to the
    actual blade geometry) with a height law tied to the real hub/shroud
    radii and blade thickness/count.

    H(s) is linearly interpolated in the normalized arc length s/S_MAX and
    measured, per ADeT's own convention (see
    ``equations.geometrical.MeridionalGeometry``), orthogonal to the local
    meridional streamline direction -- not the fixed LE/TE meridional
    angle -- so:

    .. math::
        r_{hub}(s) = r_{mean}(s) - \\tfrac{H(s)}{2} \\cos(\\varphi(s)), \\quad
        r_{tip}(s) = r_{mean}(s) + \\tfrac{H(s)}{2} \\cos(\\varphi(s))

    where :math:`\\varphi(s)` is the local tangent angle of the meridional
    path (0 deg = axial, +90 deg = radial outward). The matching axial
    offset (``z_hub_stations``/``z_tip_stations``, :math:`\\mp H(s)/2 \\sin
    \\varphi(s)`) is also computed -- at a purely radial station
    (:math:`\\varphi = \\pm 90` deg) the whole span sits in :math:`z` at a
    single radius (hub/shroud are two axial planes, not radially offset
    points), so plotting hub/tip against the *meanline's* z would show a
    null height there even though H(s) is not zero.

    The effective (blade-blockage) flow area at each station is the
    trapezoid bounded by the hub and shroud arc pitches (each
    :math:`2\\pi r / Z`, less the blade's circumferential blockage
    :math:`t / \\cos(\\beta_{bl})`, projected orthogonal to the blade
    camberline) and the two hub-to-shroud edges:

    .. math::
        A_{eff}(s) = H(s) \\left[ \\pi (r_{hub} + r_{tip})
            - Z \\, t / \\cos(\\beta_{bl}(s)) \\right]

    ``eff_area()`` below evaluates this directly (with
    :math:`\\pi(r_{hub}+r_{tip}) = 2 \\pi r_{mean}`) purely for
    inspection/plotting; the solve itself uses
    ``adet.equations.fundamental.BladeBlockage``, which computes the
    identical quantity symbolically from this class's ``r_stations`` and
    ``h_stations`` (via ``AnnulusAreas``) plus ``NumBlades``/``BldThick``/
    ``MetalAngle``, so the two never drift apart.
    """

    def __init__(
        self,
        s_stations: np.ndarray,
        r_stations: np.ndarray,
        z_stations: np.ndarray,
        mer_angle_stations: np.ndarray,
        h_in: float,
        h_out: float,
    ):
        frac = s_stations / s_stations[-1]

        self.r_stations = r_stations
        self.z_stations = z_stations
        self.mer_angle_stations = mer_angle_stations
        self.h_stations = h_in + frac * (h_out - h_in)

        # Hub/shroud are offset from the meanline along the normal to the
        # local meridional tangent (cos(phi), sin(phi)) in (z, r) -- i.e.
        # rotated +-90 deg: (-+sin(phi), +-cos(phi)). At a purely radial
        # station (phi=+-90 deg) this correctly puts the whole span in z
        # (hub/shroud are two axial planes at the same radius), not in r
        # -- see BladeChannelGeometry's docstring and the LE-height plot
        # discussion this was added for.
        r_half_span = self.h_stations / 2 * np.cos(mer_angle_stations)
        z_half_span = self.h_stations / 2 * np.sin(mer_angle_stations)
        self.r_hub_stations = r_stations - r_half_span
        self.r_tip_stations = r_stations + r_half_span
        self.z_hub_stations = z_stations + z_half_span
        self.z_tip_stations = z_stations - z_half_span

        self.area_stations = 2 * np.pi * r_stations * self.h_stations

    def eff_area(
        self, num_blades: float, blade_thickness: float, beta_bl_stations: np.ndarray
    ) -> np.ndarray:
        """Trapezoid (blade-blockage) effective flow area, Z * (hub +
        shroud arc pitches, less blade thickness) / 2 -- diagnostic only,
        see class docstring."""
        return self.area_stations - (
            self.h_stations * num_blades * blade_thickness / np.cos(beta_bl_stations)
        )


# Meanline geometry precompute (descriptive: R, H, beta_bl are boundary
# conditions of the system, not unknowns)
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
BETA_BL_STATIONS = _blade_angle_law(_s_stations / S_MAX, BETA_BL_PARAM)
DELTA_S = np.diff(_s_stations)  # streamwise spacing of each interval

# Local meridional (tangent) angle of the streamline at each station, from
# the fine Bezier path -- not just MERID_ANGLE_IN/OUT -- for
# BladeChannelGeometry's hub/tip projection.
_mer_angle_fine = np.arctan2(
    np.gradient(_meridional_path.r_coords, _arc_length),
    np.gradient(_meridional_path.z_coords, _arc_length),
)
_mer_angle_stations = np.interp(_s_stations, _arc_length, _mer_angle_fine)

_channel = BladeChannelGeometry(
    s_stations=_s_stations,
    r_stations=R_STATIONS,
    z_stations=Z_STATIONS,
    mer_angle_stations=_mer_angle_stations,
    h_in=H_IN,
    h_out=H_OUT,
)
H_STATIONS = _channel.h_stations
A_STATIONS = _channel.area_stations
R_HUB_STATIONS = _channel.r_hub_stations
R_TIP_STATIONS = _channel.r_tip_stations
Z_HUB_STATIONS = _channel.z_hub_stations
Z_TIP_STATIONS = _channel.z_tip_stations
EFF_AREA_STATIONS = _channel.eff_area(NUM_BLADES, BLADE_THICKNESS, BETA_BL_STATIONS)

# Midspan streamtube: span split into N_STREAMTUBES equal-height slices at
# every station, so the middle slice's own mean radius coincides exactly
# with R_STATIONS (the usual meanline), but its height -- and hence the
# mass flow it is assumed to carry, an equal 1/N_STREAMTUBES share -- is
# only a fraction of the full channel's. The blade-to-blade (pitch) width
# at that radius is unchanged; only the radial extent of the area used by
# MassAreaRelation/BladeBlockage shrinks accordingly (see the system build
# below, where these -- not H_STATIONS/MASS_FLOW -- become the actual BCs).
H_MIDSPAN_STATIONS = H_STATIONS / N_STREAMTUBES
EFF_AREA_MIDSPAN_STATIONS = EFF_AREA_STATIONS / N_STREAMTUBES  # diagnostic only

# ------------------------------------------------------------
# Spanwise mass-flow redistribution (Van den Braembussche, *Design and
# Analysis of Centrifugal Compressors*, 2019, Ch. 3 "Radial Impeller Flow
# Calculation", Sec. 3.1.1, Eqs. 3.3-3.6): radial equilibrium + rothalpy
# conservation across the span make the meridional velocity vary from hub
# to shroud with the local meridional-streamline curvature -- splitting
# the mass flow evenly across the N_STREAMTUBES streamtubes above ignores
# this and is an algebraic no-op (H and mf both scale by the same 1/N,
# cancelling in V_mer = mf/(rho*A)). For purely axial/radial blades
# (beta=0, this rotor's LE/TE limits), Eq. (3.4) gives
#     d(Wm)/dn = Wm / Rn
# marched hub -> tip across the streamtubes (Eq. 3.5):
#     Wm_j = Wm_{j-1} (1 + dn / Rn)
# with Rn the local meridional-path curvature radius (only resolved
# streamwise here, like beta_bl/H -- a full spanwise curvature field needs
# actual hub/shroud geometry this model doesn't carry) and dn = H(s) /
# N_STREAMTUBES. The hub-streamtube starting value is set so the spanwise
# sum of streamtube mass flows matches continuity (Eq. 3.6), assuming
# uniform density across span at each station (a full spanwise density
# profile needs a true spanwise solve, out of scope for this meanline
# model) -- so only the *relative* (Wm_j * r_j) weighting matters below,
# not the absolute Wm level.
# ------------------------------------------------------------
USE_SPANWISE_MASS_REDISTRIBUTION = True
MIDSPAN_INDEX = N_STREAMTUBES // 2

_dz_ds = np.gradient(_meridional_path.z_coords, _arc_length)
_dr_ds = np.gradient(_meridional_path.r_coords, _arc_length)
_d2z_ds2 = np.gradient(_dz_ds, _arc_length)
_d2r_ds2 = np.gradient(_dr_ds, _arc_length)
# Signed curvature of the meridional path; 1/curvature = Rn. Positive
# where the path curves toward the shroud-offset direction
# (-sin(phi), cos(phi)), see BladeChannelGeometry's tip offset.
_curvature_fine = (_dz_ds * _d2r_ds2 - _dr_ds * _d2z_ds2) / np.power(
    _dz_ds**2 + _dr_ds**2, 1.5
)
CURVATURE_RADIUS_STATIONS = 1.0 / np.interp(_s_stations, _arc_length, _curvature_fine)


def _spanwise_mass_fraction(
    h_stations, r_stations, mer_angle_stations, curvature_radius_stations, n_sub
):
    """Fraction of the total mass flow carried by streamtube
    MIDSPAN_INDEX (of n_sub equal-height streamtubes) at every station,
    from Eqs. (3.4)-(3.6) -- see the block comment above."""
    dn = h_stations / n_sub
    j = np.arange(n_sub)
    # Shape factor (Eq. 3.5) relative to the hub-most streamtube, assuming
    # constant curvature across the (thin) span at each station.
    shape = (1.0 + dn[:, None] / curvature_radius_stations[:, None]) ** j[None, :]
    # Streamtube-center radius, offset from the hub by (j+1/2) steps along
    # the same spanwise normal BladeChannelGeometry uses for r_hub/r_tip.
    r_hub = r_stations - h_stations / 2 * np.cos(mer_angle_stations)
    r_center = r_hub[:, None] + (j[None, :] + 0.5) * dn[:, None] * np.cos(
        mer_angle_stations[:, None]
    )
    weight = shape * r_center  # ~ rho * Wm * r * dn, rho*dn common to all j
    fraction = weight / weight.sum(axis=1, keepdims=True)
    return fraction[:, MIDSPAN_INDEX]


MASS_FLOW_FRACTION_STATIONS = _spanwise_mass_fraction(
    H_STATIONS,
    R_STATIONS,
    _mer_angle_stations,
    CURVATURE_RADIUS_STATIONS,
    N_STREAMTUBES,
)
MASS_FLOW_MIDSPAN_STATIONS = (
    MASS_FLOW * MASS_FLOW_FRACTION_STATIONS
    if USE_SPANWISE_MASS_REDISTRIBUTION
    else np.full(N_STATIONS, MASS_FLOW_MIDSPAN)
)

# Node and custom variable declarations
nodes = [NodeVariables(i) for i in range(N_STATIONS)]

# Local node placeholders (upstream/downstream) for residual() hints
n0 = NodeVariables(0)
n1 = NodeVariables(1)


class BladeLoadingVariables(VariableEnum):
    """Custom variables for the blade-to-blade loading (Section 3.1.2)."""

    # Floored at 5 m/s: near the LE, the inviscid loading law (Eq. 3.13)
    # can drive W_ps through zero/negative (a modeling artifact).
    W_ss = VarSpec('W_ss', 'm / s', 150.0, (5.0, 2e3))
    W_ps = VarSpec('W_ps', 'm / s', 100.0, (5.0, 2e3))
    # Lower bound raised to 10 m/s (from a symmetric +-1e3): DeltaW = W_SS -
    # W_PS (Eq. 3.13), and IPOPT can otherwise converge onto the unphysical
    # branch where the sign flips (W_PS > W_SS, p_SS > p_PS -- exactly the
    # swap bug diagnosed earlier). Bounding it here keeps the search inside
    # the physically feasible region instead of only fixing symptoms after
    # the fact; only affects interior stations, since LE/TE DeltaW is fixed
    # via boundary condition (USE_KUTTA_LE/TE) and excluded from bounds.
    DeltaW = VarSpec('delta_W', 'm / s', 20.0, (10.0, 1e3))  # Eq. (3.13)
    P_ss = VarSpec('p_ss', 'Pa', 2e5, (1e2, 2e7))
    P_ps = VarSpec('p_ps', 'Pa', 2.5e5, (1e2, 2e7))


bl_nodes = [BladeLoadingVariables(i) for i in range(N_STATIONS)]
bl0 = BladeLoadingVariables(0)
bl1 = BladeLoadingVariables(1)

TRAILING_EDGE = N_STATIONS - 1

# Transition point s* (Eqs. 3.9-3.12): ZeroDeviation up to here, then
# SlipTransition's quadratic blend down to zero loading at the exducer.
TRANSITION_STAR = int(round(TRANSITION_FRAC_TE * TRAILING_EDGE))

# Inlet-side counterpart: the LE flow angle is a free unknown (slip-driven
# incidence, generally off the blade angle), while ZeroDeviation forces
# beta = beta_bl exactly at the very next station -- that one-interval jump
# spikes BladeToBladeLoading's turning_term/DeltaW right after the LE.
# Mirror the exducer treatment: blend from the free LE value onto the
# blade-angle law over a short inlet region (same SlipTransition quadratic,
# s_end=0 and beta_slip1 bound to node 0 instead of the trailing edge).
TRANSITION_STAR_IN = max(1, int(round(TRANSITION_FRAC_LE * TRAILING_EDGE)))

# Local slope of the (possibly nonlinear, see BETA_BL_PARAM) blade-angle
# law at each transition's anchor point, for SlipTransition's
# slope-matching.
_BETA_BL_GRAD = np.gradient(BETA_BL_STATIONS, _s_stations)

# Effective LE blend range: stations 1..LE_BLEND_END use SlipTransition;
# collapses to none (sharp ZeroDeviation jump right at station 1, the
# original pre-fix behavior) when USE_SLIP_TRANSITION_LE is off.
LE_BLEND_END = TRANSITION_STAR_IN if USE_SLIP_TRANSITION_LE else 0

# The TE blend only makes sense when the TE flow angle is actually free
# (USE_KUTTA_TE on) -- otherwise there is no downstream value left to blend
# onto, so ZeroDeviation is extended straight through to TRAILING_EDGE
# instead (see CarryLoadingToLE's docstring for the LE-side analog).
LAST_FIXED_BETA_STATION = TRAILING_EDGE - 1 if USE_KUTTA_TE else TRAILING_EDGE
TE_BLEND_START = (
    TRANSITION_STAR + 1
    if (USE_SLIP_TRANSITION_TE and USE_KUTTA_TE)
    else LAST_FIXED_BETA_STATION + 1
)


# ============================================================
# 2. Custom equation classes
#    (identical physics to radial_impeller_blade_loading.py -- rothalpy
#    conservation, blade-to-blade loading, slip transition and SS/PS
#    velocity/pressure recovery don't depend on the radius direction)
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
    """Blade-to-blade SS-PS velocity difference (Van den Braembussche
    Eq. 3.13), backward finite difference between node 0 (upstream) and
    node 1 (downstream), spacing ``delta_s``."""

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

        # Sign flip vs. the compressor case: for radially-inward flow, the
        # Coriolis contribution (rotation_term) acts on the opposite face
        # relative to a radially-outward compressor, so only its sign
        # flips here -- turning_term (blade-camber loading) keeps the same
        # convention as radial_impeller_blade_loading.py regardless of flow
        # direction. Flipping the whole bracket instead (as before) also
        # negated turning_term, which happened to still look right for the
        # R-decreasing case (both terms negative there) but silently swaps
        # W_ss/W_ps, p_ss/p_ps for R=const (axial) geometries, where
        # rotation_term vanishes and turning_term is all that's left.
        loading = -pitch_minus_thickness * (rotation_term + turning_term)

        return delta_w1 - loading


class SlipTransition(EquationBase):
    """Stand-in for the flow-angle/blade-angle deviation model of
    Eqs. (3.9)-(3.12): a quadratic matching the blade-angle law's value and
    slope at one anchor station, passing exactly through a second
    (generally free) flow angle elsewhere. Used near the LE (blending the
    slip incidence onto the blade-angle law) and near the TE (blending onto
    the Kutta-pinned value); ``ZeroDeviation`` holds in between."""

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


class CarryLoadingToLE(EquationBase):
    """Alternative to the zero-loading LE boundary condition
    (USE_KUTTA_LE=False): no equation otherwise ties bl_nodes[0].DeltaW to
    anything (no pair exists upstream of the LE), so instead carry the
    first interval's loading back to the LE (zero-gradient extrapolation)
    rather than assuming no loading has developed yet."""

    def residual(self, delta_w0: bl0.DeltaW.Hint, delta_w1: bl1.DeltaW.Hint):
        return delta_w0 - delta_w1


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
    """Static pressure on the SS/PS blade surfaces, from rothalpy
    conservation + the isentropic (inviscid) assumption shared by both
    surfaces."""

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
# 3. System build
# ============================================================
system = CasadiSystem(num_span=1)

system.fluid_settings = FluidSettings(
    fluid_state=IdealGasState(gamma=1.4, gas_constant=287.0, viscosity=1.8e-5),
    update_variables=(thrm.Pressure, thrm.Temperature),
)

EQUATIONS = {}
for i in range(N_STATIONS):
    EQUATIONS[Kinematics()] = i
    if i == 0:
        # Rotor LE: flow-slip-driven incidence (Chen & Baines 1994)
        # instead of zero-incidence.
        EQUATIONS[OptimalIncidenceRadialInflowTurbine()] = i
    elif i <= LE_BLEND_END:
        EQUATIONS[
            SlipTransition(
                s_i=_s_stations[i],
                s_star=_s_stations[TRANSITION_STAR_IN],
                s_end=0.0,
                beta_bl_star=BETA_BL_STATIONS[TRANSITION_STAR_IN],
                slope_star=_BETA_BL_GRAD[TRANSITION_STAR_IN],
            )
        ] = (i, 0)
    elif i <= LAST_FIXED_BETA_STATION and i < TE_BLEND_START:
        EQUATIONS[ZeroDeviation()] = i
    elif i <= LAST_FIXED_BETA_STATION:
        EQUATIONS[
            SlipTransition(
                s_i=_s_stations[i],
                s_star=_s_stations[TRANSITION_STAR],
                s_end=S_MAX,
                beta_bl_star=BETA_BL_STATIONS[TRANSITION_STAR],
                slope_star=_BETA_BL_GRAD[TRANSITION_STAR],
            )
        ] = (i, TRAILING_EDGE)
    # Else (i > LAST_FIXED_BETA_STATION, only possible for i ==
    # TRAILING_EDGE when USE_KUTTA_TE is on): flow angle left free, pinned
    # by the Kutta condition instead (see BOUNDARY_CONDITIONS).
    EQUATIONS[AnnulusAreas()] = i
    EQUATIONS[BladeBlockage()] = i
    EQUATIONS[MassAreaRelation()] = i
    EQUATIONS[TotalStaticMatching()] = i
    EQUATIONS[SuctionPressureVelocities()] = i
    EQUATIONS[BladeSurfacePressures()] = i

for i in range(1, N_STATIONS):
    # MassConservation (mf_i = mf_{i-1}) only applies when the midspan
    # mass flow is a single constant carried unchanged along s; with
    # USE_SPANWISE_MASS_REDISTRIBUTION it instead varies station to
    # station (curvature redistributes flow across span), so it is
    # prescribed directly as a boundary condition at every node below
    # instead (removing exactly as many free StreamMassFlow unknowns as
    # equations dropped here, so the system stays square).
    if not USE_SPANWISE_MASS_REDISTRIBUTION:
        EQUATIONS[MassConservation()] = (i - 1, i)
    EQUATIONS[IsentropicLink()] = (i - 1, i)
    EQUATIONS[RothalpyConservation()] = (i - 1, i)
    EQUATIONS[BladeToBladeLoading(delta_s=DELTA_S[i - 1])] = (i - 1, i)

if not USE_KUTTA_LE:
    EQUATIONS[CarryLoadingToLE()] = (0, 1)

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

BOUNDARY_CONDITIONS = {}
for i in range(N_STATIONS):
    n = nodes[i]
    BOUNDARY_CONDITIONS[n.geo.RDistr] = R_STATIONS[i]
    BOUNDARY_CONDITIONS[n.geo.HDistr] = H_MIDSPAN_STATIONS[i]
    BOUNDARY_CONDITIONS[n.geo.MetalAngle] = BETA_BL_STATIONS[i]
    BOUNDARY_CONDITIONS[n.geo.NumBlades] = NUM_BLADES
    BOUNDARY_CONDITIONS[n.geo.BldThick] = BLADE_THICKNESS
    # No boundary-layer model in this inviscid blade-to-blade example, so
    # BladeBlockage's displacement-thickness term is zero (blade-thickness
    # blockage only).
    BOUNDARY_CONDITIONS[n.oth.DispThick] = 0.0
    BOUNDARY_CONDITIONS[n.kin.Omega] = OMEGA
    # Midspan streamtube's share of the mass flow -- a single constant
    # propagated by MassConservation, or (USE_SPANWISE_MASS_REDISTRIBUTION)
    # prescribed station-by-station from the curvature-driven spanwise
    # redistribution (Eqs. 3.4-3.6; see MASS_FLOW_MIDSPAN_STATIONS above).
    if USE_SPANWISE_MASS_REDISTRIBUTION or i == 0:
        BOUNDARY_CONDITIONS[n.oth.StreamMassFlow] = MASS_FLOW_MIDSPAN_STATIONS[i]

# Inlet total thermodynamic state (node 0 only)
BOUNDARY_CONDITIONS[nodes[0].tot.Pressure] = Quantity(P0_TOT, 'Pa')
BOUNDARY_CONDITIONS[nodes[0].tot.Temperature] = Quantity(T0_TOT, 'K')

# Cone angle for OptimalIncidenceRadialInflowTurbine (purely radial inlet)
BOUNDARY_CONDITIONS[nodes[0].geo.MeridionalAngle] = MERID_ANGLE_IN

# Zero incidence at the radial inlet: no blade loading yet (off ->
# CarryLoadingToLE closes bl_nodes[0].DeltaW instead, see USE_KUTTA_LE).
if USE_KUTTA_LE:
    BOUNDARY_CONDITIONS[bl_nodes[0].DeltaW] = 0.0

# Kutta condition at the exducer exit: W_SS = W_PS (off -> the flow angle
# is pinned to the blade-angle law there instead, via LAST_FIXED_BETA_STATION,
# and BladeToBladeLoading's (N-2, N-1) equation closes DeltaW as usual).
if USE_KUTTA_TE:
    BOUNDARY_CONDITIONS[bl_nodes[TRAILING_EDGE].DeltaW] = 0.0

system.add_boundary_conditions(BOUNDARY_CONDITIONS)

system.build()

# ============================================================
# 4. Solve
# ============================================================
# The exducer turns the flow by ~58 deg over the meanline; the uniform
# fallback guess is too far from that for IPOPT. Seed the flow angle with
# the blade metal angle law and W with the local blade speed.
_RHO0_APPROX = P0_TOT / (287.0 * T0_TOT)  # crude ideal-gas density estimate

MANUAL_GUESSES = {}
for i in range(N_STATIONS):
    MANUAL_GUESSES[nodes[i].kin.FlowAngleRel] = BETA_BL_STATIONS[i]
    MANUAL_GUESSES[nodes[i].kin.W_mag] = max(OMEGA * R_STATIONS[i], 50.0)
    MANUAL_GUESSES[nodes[i].oth.SlipFactor] = 0.9
    # Seed mass flow/density/meridional velocity consistently (mf = rho *
    # Vm * A, midspan streamtube's share/area): otherwise IPOPT's guess
    # violates MassConservation and gets stuck near the inlet.
    MANUAL_GUESSES[nodes[i].oth.StreamMassFlow] = MASS_FLOW_MIDSPAN_STATIONS[i]
    MANUAL_GUESSES[nodes[i].stc.Density] = _RHO0_APPROX
    MANUAL_GUESSES[nodes[i].kin.V_mer] = MASS_FLOW_MIDSPAN_STATIONS[i] / (
        _RHO0_APPROX * 2 * np.pi * R_STATIONS[i] * H_MIDSPAN_STATIONS[i]
    )

# Rotor LE guessed near the typical empirical incidence, not the blade
# angle, since OptimalIncidenceRadialInflowTurbine lets it deviate.
MANUAL_GUESSES[nodes[0].kin.FlowAngleRel] = np.radians(-20.0)

x0 = system.get_guess(MANUAL_GUESSES, fallback=0.6)
kn = system.get_boundary_conds()

# Bound the free-deviation regions' flow angle within +-25 deg of the
# blade metal angle law (covers both the inlet-side SlipTransition
# stations and, when USE_KUTTA_TE is on, the exit-side ones -- including
# the free TRAILING_EDGE station itself), to keep the search away from
# unphysical values.
CUSTOM_BOUNDS = {}
slack = np.radians(25.0)
_bound_indices = list(range(1, LE_BLEND_END + 1))
if USE_KUTTA_TE:
    _bound_indices += list(range(TE_BLEND_START, N_STATIONS))
for i in _bound_indices:
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

# Polish with a second, tighter bounded IPOPT solve rather than the usual
# unconstrained KINSOL pass: KINSOL ignores bounds and would march back to
# the unphysical negative-W_ps root the W_ss/W_ps floor exists to avoid.
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
# 5. Post-processing
# ============================================================
s_over_smax = _s_stations / S_MAX

w_mean = np.array([sol_dict[nodes[i].kin.W_mag][0] for i in range(N_STATIONS)])
w_ss = np.array([sol_dict[bl_nodes[i].W_ss][0] for i in range(N_STATIONS)])
w_ps = np.array([sol_dict[bl_nodes[i].W_ps][0] for i in range(N_STATIONS)])
p_ss = np.array([sol_dict[bl_nodes[i].P_ss][0] for i in range(N_STATIONS)])
p_ps = np.array([sol_dict[bl_nodes[i].P_ps][0] for i in range(N_STATIONS)])
p_mean = np.array([sol_dict[nodes[i].stc.Pressure][0] for i in range(N_STATIONS)])
beta_flow = np.degrees(
    np.array([sol_dict[nodes[i].kin.FlowAngleRel][0] for i in range(N_STATIONS)])
)
beta_metal = np.degrees(BETA_BL_STATIONS)

# Diagnostic: R*Vu should decrease along the meanline, so Euler work
# Omega * d(R*Vu) is negative -- work extracted from the fluid.
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

fig, axs = plt.subplots(1, 5, figsize=(25, 5))

# --- Meridional profile: meanline plus hub/shroud from BladeChannelGeometry
axs[0].plot(_meridional_path.z_coords, _meridional_path.r_coords, 'k-', label='mean')
axs[0].plot(Z_STATIONS, R_STATIONS, 'o', color='tab:red', ms=4)
axs[0].plot(Z_HUB_STATIONS, R_HUB_STATIONS, '-o', color='tab:blue', ms=3, label='hub')
axs[0].plot(
    Z_TIP_STATIONS, R_TIP_STATIONS, '-o', color='tab:green', ms=3, label='shroud'
)
axs[0].set_aspect('equal')
axs[0].set_xlabel(r'$z$ / [m]')
axs[0].set_ylabel(r'$r$ / [m]')
axs[0].set_title('Meridional path (hub/mean/shroud)')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# --- Flow area: full channel (raw annulus / blade-blockage effective) vs.
# the midspan streamtube actually fed into the solve (1/N_STREAMTUBES of
# the effective area, see H_MIDSPAN_STATIONS)
axs[1].plot(s_over_smax, A_STATIONS * 1e4, '--', color='k', label='annulus (raw)')
axs[1].plot(
    s_over_smax,
    EFF_AREA_STATIONS * 1e4,
    '-o',
    color='tab:purple',
    label='effective (full)',
)
axs[1].plot(
    s_over_smax,
    EFF_AREA_MIDSPAN_STATIONS * 1e4,
    '-o',
    color='tab:orange',
    label=f'effective (midspan, 1/{N_STREAMTUBES})',
)
axs[1].set_xlabel(r'$s / s_{max}$')
axs[1].set_ylabel(r'$A$ / [cm$^2$]')
axs[1].set_title('Flow area distribution')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# --- Velocity distribution
axs[2].plot(s_over_smax, w_ss, '-o', label=r'$W_{SS}$')
axs[2].plot(s_over_smax, w_mean, '--', color='k', label=r'$\widetilde{W}$')
axs[2].plot(s_over_smax, w_ps, '-o', label=r'$W_{PS}$')
axs[2].set_xlabel(r'$s / s_{max}$')
axs[2].set_ylabel(r'$W$ / [m/s]')
axs[2].set_title('Blade-to-blade velocity distribution')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

# --- Pressure distribution
axs[3].plot(s_over_smax, p_ss / 1e5, '-o', label=r'$p_{SS}$')
axs[3].plot(s_over_smax, p_mean / 1e5, '--', color='k', label=r'$p$ (mean)')
axs[3].plot(s_over_smax, p_ps / 1e5, '-o', label=r'$p_{PS}$')
axs[3].set_xlabel(r'$s / s_{max}$')
axs[3].set_ylabel(r'$p$ / [bar]')
axs[3].set_title('Blade surface pressure distribution')
axs[3].legend()
axs[3].grid(True, alpha=0.3)

# --- Relative flow angle vs. blade metal angle
axs[4].plot(s_over_smax, beta_metal, '--', color='k', label=r'$\beta_{bl}$ (metal)')
axs[4].plot(s_over_smax, beta_flow, '-o', color='tab:blue', label=r'$\beta$ (flow)')
axs[4].set_xlabel(r'$s / s_{max}$')
axs[4].set_ylabel(r'$\beta$ / [deg]')
axs[4].set_title(f'Flow angle vs. blade metal angle ({BETA_BL_PARAM})')
axs[4].legend()
axs[4].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
