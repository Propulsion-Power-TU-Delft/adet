"""
Sundstrand T-100 MPSPU radial-inflow turbine (Jones, 1996 / Sauret, 2012)

    A. C. Jones, "Design and Test of a Small, High Pressure Ratio Radial
    Turbine," ASME J. Turbomach., vol. 118, pp. 362-370, April 1996.

    E. Sauret, "Open Design of High Pressure Ratio Radial-Inflow Turbine
    for Academic Validation," Proc. ASME IMECE2012, IMECE2012-88315.

Jones (1996) gives the cycle boundary conditions (Table 2) and mean-line
velocity triangles (Fig. 4);  geometry is only available as scanned
drawings. Sauret (2012) independently tabulates the exact 3D geometry of
this same turbine ( Tables 1-2 and Appendix A), so the geometry used here
is taken from Sauret wherever available, falling back to Jones' only
for the few items Sauret doesn't tabulate (nozzle inlet swirl angle).

Rotor exit static pressure is taken from Sauret's Table 4
"Present" (RITAL) result at engine conditions.

State:
------
- Design point only (50 hp rating)
- No exhaust diffuser
- Zero-deviation at every row exit
- Currently not running without initial guess
"""

import logging
import math
import shutil
from pathlib import Path

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity
from tabulate import tabulate

from adet.assemblers import CasadiSystem
from adet.components.blade_row import BladeRow, Interspace, RowGeometry
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.base_equation import LossApplier
from adet.equations.definitions import BoundaryLayerRatios, IsentropicProperties
from adet.equations.fundamental import IdealInterspace, InterspaceLoss
from adet.equations.nondimensional import (
    GammaPV,
    TotalStaticExpansionEfficiency,
    TotalTotalExpansionEfficiency,
)
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.losses.mixing import (
    DentonBaumgartnerMixingLoss,
    SieverdingBasePressure,
)
from adet.losses.rit import (
    ChenSlip,
    EndwallLoss,
    ImpellerPassageLoss,
    StatorProfileLoss,
)
from adet.losses.rit_sauret import (
    SauretClearanceLoss,
    SauretIncidenceLoss,
    SauretWindageLoss,
)
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_velocity_triangles, setup_mpl
from adet.variables import KinematicVariables, NodeVariables
from adet.varspec import DEF_NODE

logger = logging.getLogger(__name__)
setup_logger(logger)
setup_mpl(
    {
        'font.family': ['sans-serif'],
        'font.size': 20,
        'axes.labelsize': 18,
        'legend.fontsize': 18,
    }
)

# Paper figures: saved as PDF at the printed width. Text is typeset by LaTeX
# through the pgf backend (pdflatex), so the labels match the manuscript body.
# (matplotlib's usetex route needs type1cm/type1ec, which this TeX install lacks.)
FIG_DIR = Path('/home/francesco/Documents/aerx_springer/figures/rotating_components')
DATA_DIR = Path('/home/francesco/Documents/aerx_springer/local/scripts/adet_rit_validation/results')
SAVE_FIGURES = True
if SAVE_FIGURES and not shutil.which('pdflatex'):
    raise RuntimeError('SAVE_FIGURES needs pdflatex (pgf backend)')
if SAVE_FIGURES:
    plt.rcParams.update(
        {
            'pgf.texsystem': 'pdflatex',
            'pgf.rcfonts': False,
            'pgf.preamble': r'\usepackage{amsmath}',
            'font.size': 9,
            'axes.labelsize': 9,
            'axes.titlesize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
        }
    )


def save_pdf(fig, path: Path) -> None:
    """Save a figure as a LaTeX-typeset PDF via the pgf backend."""
    from matplotlib.backends.backend_pgf import FigureCanvasPgf

    fig.canvas = FigureCanvasPgf(fig)
    fig.savefig(path)

# *** Global settings
PLOTS = True
USE_SLIP_MODEL = False
RUN_SPEEDLINE = True
RUN_PR_SWEEP = False

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)
n4 = NodeVariables(4)
n5 = NodeVariables(5)


class AddImpellerLosses(LossApplier):
    """Apply the rotor's passage + leakage + incidence + windage + endwall +
    mixing losses to the static entropy rise."""

    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
        ds_profile1: n1.loss.Ds_profile.Hint,
        ds_leakage1: n1.loss.Ds_leakage.Hint,
        ds_incidence1: n1.loss.Ds_incidence.Hint,
        # ds_endwall1: n1.loss.Ds_endwall.Hint,
        ds_windage1: n1.loss.Ds_secondary.Hint,
        ds_mixing1: n1.loss.Ds_mixing.Hint,
    ):
        return s1 - (
            s0 + ds_profile1 + ds_mixing1 + ds_leakage1 + ds_incidence1 + ds_windage1
        )


class AddStatorLosses(LossApplier):
    """Apply the nozzle's profile + endwall + mixing losses to the static
    entropy rise."""

    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
        ds_profile1: n1.loss.Ds_profile.Hint,
        ds_endwall1: n1.loss.Ds_endwall.Hint,
        ds_mixing1: n1.loss.Ds_mixing.Hint,
    ):
        return s1 - (s0 + ds_profile1 + ds_endwall1 + ds_mixing1)


# |> Machine stations, in the order the components are chained below (each
# component owns two fresh global nodes: inlet, outlet)

STATION_LABELS = {
    0: 'Nozzle inlet',
    1: 'Nozzle outlet',
    2: 'Interspace inlet',
    3: 'Interspace outlet',
    4: 'Rotor inlet',
    5: 'Rotor outlet',
}


inl = Inlet(
    {
        n0.tot.Pressure: Quantity(580.4, 'kPa'),  # Sauret Table 1
        n0.tot.Temperature: Quantity(1056.5, 'K'),  # Sauret Table 1
        # n0.oth.TotMassFlow: 0.33,  # kg/s, Sauret Table 1
        # NOTE (assumed / Jones-only): Sauret doesn't tabulate nozzle inlet
        # swirl; Jones gives a 20-40 deg range, midpoint used here.
        n0.kin.FlowAngleRel: Quantity(30.0, 'deg'),
        n0.geo.Height: Quantity(0.00635, 'm'),  # nozzle inlet height (Sauret Table 2)
        n0.geo.MeridionalAngle: Quantity(-90, 'deg'),
    }
)

casing = Shaft(Quantity(0, 'rpm'), is_constrained=True)
shaft = Shaft(Quantity(106588, 'rpm'), is_constrained=True)  # Sauret Table 1


def build_network(include_losses: bool) -> ComponentNetwork:
    """Build the nozzle/interspace/rotor network.

    With ``include_losses=False`` every entropy-generating loss mechanism is
    replaced by an ``IsentropicLink`` (s_out = s_in), giving a much
    easier-to-solve system whose solution is used as the initial guess for
    the full, lossy system (see below).
    """
    stator_extra_equations: dict = {}  # No incidence
    rotor_extra_equations: dict = {}
    if USE_SLIP_MODEL:
        rotor_extra_equations[ChenSlip()] = 0

    if include_losses:
        stator_extra_equations.update(
            {
                AddStatorLosses(): (0, 1),
                StatorProfileLoss(): (0, 1),
                EndwallLoss(): (0, 1),
                IsentropicProperties(): (0, 1),
                BoundaryLayerRatios(): 1,
                GammaPV(): 0,
                GammaPV(): 1,
                SieverdingBasePressure(): (0, 1),
                DentonBaumgartnerMixingLoss(): 1,
            }
        )
        # Zero deviation at the stator exit (the rotor gets it below)
        stator_extra_equations[ZeroDeviation()] = 1

        rotor_extra_equations.update(
            {
                AddImpellerLosses(): (0, 1),
                IsentropicProperties(): (0, 1),
                ImpellerPassageLoss(0.1): (0, 1),
                SauretClearanceLoss(): (0, 1),
                # ImpellerIncidenceLoss(-0.94): (0, 1),
                SauretIncidenceLoss(): (0, 1),
                # EndwallLoss(): (0, 1),
                BoundaryLayerRatios(): 1,
                SauretWindageLoss(): (0, 1),
                GammaPV(): 0,
                GammaPV(): 1,
                SieverdingBasePressure(): (0, 1),
                DentonBaumgartnerMixingLoss(): 1,
            }
        )
    else:
        stator_extra_equations[IsentropicLink()] = (0, 1)
        rotor_extra_equations[IsentropicLink()] = (0, 1)

    stator = BladeRow(
        'nozzle',
        bound_cond={
            n0.geo.Rmid: Quantity(0.074, 'm'),  # nozzle inlet radius
            n1.geo.Height: Quantity(0.006, 'm'),  # nozzle exit height (Sauret Table 2)
            n1.geo.Rmid: Quantity(0.0635, 'm'),  # nozzle exit radius (Sauret Table 2)
            n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
            n0.geo.MetalAngle: Quantity(30, 'deg'),  # Sauret Fig. 8(a)
            n1.geo.MetalAngle: Quantity(78.1, 'deg'),  # Sauret Fig. 8(a)
            n0.geo.BldThick: Quantity(0.661, 'mm'),  # Sauret Table 6 (m=0%)
            n1.geo.BldThick: Quantity(0.51, 'mm'),  # Sauret Table 2
            n1.geo.NumBlades: 19,  # Sauret Table 1
            n1.geo.MerChord: Quantity(0.1, 'mm'),  # No influence
            n1.oth.MomByBld: 0.1,
            n1.oth.DispByMom: 2.0,
            n1.oth.DispByHgt: 0.15,
        },
        shaft=casing,
        extra_equations=stator_extra_equations,
    )

    interspace_eq = InterspaceLoss() if include_losses else IdealInterspace()

    interspace = Interspace(
        'intrspc',
        {
            n1.geo.Rmid: Quantity(0.0582, 'm'),  # rotor inlet tip radius
            n1.geo.Height: Quantity(0.00635, 'm'),  # vaneless space height
        },
        extra_equations={
            interspace_eq: (0, 1),
        },
    )

    rotor = BladeRow(
        'impeller',
        bound_cond={
            # n1.geo.Height: Quantity(0.03683 - 0.01524, 'm'),  # exducer tip - hub
            n1.geo.Rtip: Quantity(0.03683, 'm'),
            n1.geo.Rhub: Quantity(0.01524, 'm'),
            n1.geo.MeridionalAngle: Quantity(0, 'deg'),  # axial exducer exit
            n0.geo.MetalAngle: Quantity(0, 'deg'),  # purely radial LE blades
            n1.geo.MetalAngle: Quantity(-51.628, 'deg'),  # RMS (hub/tip)
            n0.geo.BldThick: Quantity(0.86, 'mm'),  # Sauret Table 2 / Appendix
            n1.geo.BldThick: Quantity(0.76, 'mm'),  # Sauret Table 2 / Appendix
            n1.geo.AxialLength: Quantity(0.0389, 'm'),  # axial length, Sauret Table 2
            n1.geo.MerChord: Quantity(0.0389, 'm'),  # axial length, Sauret Table 2
            n1.geo.NumBlades: 16,  # Sauret Table 1 / Jones text
            n0.geo.TipClearance: Quantity(0.4, 'mm'),  # Sauret Table 2
            n1.geo.TipClearance: Quantity(0.23, 'mm'),  # Sauret Table 2
            n1.oth.MomByBld: 0.075,
            n1.oth.DispByMom: 2.0,
            n1.oth.DispByHgt: 0.09,
            # *** OUTLET
            n1.stc.Pressure: Quantity(94.7, 'kPa'),  # Sauret Table 4
            # n1.kin.RelMach: 0.713,  # Sauret Table 4
            # n1.kin.FlowAngleAbs: 0.0,  # Sauret Table 4
        },
        shaft=shaft,
        extra_equations={
            **rotor_extra_equations,
            ZeroDeviation(): 1,
        },
    )

    if USE_SLIP_MODEL:
        # ``ChenSlip`` sets the rotor inlet tangential velocity
        rotor._from_prev_node.discard(KinematicVariables(DEF_NODE).V_tan)

    abs_state = DebugAbstractState('HEOS', 'Air')

    fluid_settings = FluidSettings(
        fluid_state=abs_state,
        update_variables=(
            n0.stc.Pressure.Glob,
            n0.stc.Temperature.Glob,
        ),
    )

    ntw = ComponentNetwork(
        fluid_settings,
        inl,
        CasadiSystem(1),
        [
            stator,
            interspace,
            rotor,
        ],
    )

    ntw.system.add_equation(TotalStaticExpansionEfficiency(), (0, 5))
    ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 5))

    return ntw


# ============================================================
# Bounds -- the isentropic initialization and the full, lossy system are
# both seeded from adet's own default guesses (no hand-picked values); only
# bounds are supplied to keep the solve on the physical branch.
# ============================================================
BOUNDS = {
    n1.kin.Mach: (0.3, 1.05),
    n0.stc.Temperature.Glob: (200, 1500),
    n0.stc.Pressure.Glob: (1, 2e6),
    n0.stc.Entropy.Glob: (3000.0, 7000.0),
}

# ============================================================
# Stage 1: isentropic initialization (no loss models)
# starting point for the full, lossy system below.
# ============================================================
ntw_isen = build_network(include_losses=False)
ntw_isen.build()

rtfn_isen = ntw_isen.system.make_rootfinder('ipopt', opts={'error_on_fail': False})
x0_isen = ntw_isen.system.get_guess(fallback=0.5)
kn_isen = ntw_isen.system.get_boundary_conds()
bnd_isen = ntw_isen.system.get_bounds(BOUNDS)

sol_isen = solve_root_problem(
    rtfn_isen, x0_isen, kn_isen, bnd_isen, suppress_output=False
)
rtfn_isen = ntw_isen.system.make_rootfinder('kinsol')
sol_isen = solve_root_problem(rtfn_isen, sol_isen, kn_isen, bnd_isen)

isen_data = ntw_isen.system.sol_to_dict(sol_isen)
logger.info(
    'Isentropic initialization solved; seeding the full (lossy) system with it.'
)

# Guesses for isentropic properties
isen_property_guess = {}
for _n in (n1, n5):
    isen_property_guess.update(
        {
            _n.oth.Tis_tot: isen_data[_n.tot.Temperature],
            _n.oth.Enthalpy_totIs: isen_data[_n.tot.Enthalpy],
            _n.oth.Tis_stc: isen_data[_n.stc.Temperature],
            _n.oth.Enthalpy_Is: isen_data[_n.stc.Enthalpy],
        }
    )

# ============================================================
# Stage 2: full system, with loss models, seeded from the isentropic
# solution above instead of the hand-tuned guess.
# ============================================================
ntw = build_network(include_losses=True)
ntw.build()

rtfn = ntw.system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False},
)

x0 = ntw.system.get_guess({**isen_data, **isen_property_guess}, fallback=0.5)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds(BOUNDS)

sol = solve_root_problem(rtfn, x0, kn, suppress_output=False)

# Kinsol pass
rtfn_k = ntw.system.make_rootfinder('kinsol')
rtfn_n = ntw.system.make_rootfinder('newton')
# sol = solve_root_problem(rtfn_k, sol, kn)

rtfn = ntw.system.make_rootfinder('ipopt', {'error_on_fail': False})

sol_data = ntw.system.sol_to_dict(sol)

# ============================================================
# U/a0 sweep: the rotor shaft speed is a fixed boundary condition, so the
# solved network is reused. Only the omega entry of the boundary-condition
# vector is rescaled per point, warm-started from the previous solution.
# ============================================================
R_ROTOR_INLET = 0.0582  # m, rotor inlet tip radius (used for U)

LOSS_MECHANISMS = {
    'Nozzle': [
        ('Profile', lambda n: n.loss.Ds_profile),
        ('Endwall', lambda n: n.loss.Ds_endwall),
        ('Mixing', lambda n: n.loss.Ds_mixing),
    ],
    'Rotor': [
        ('Profile', lambda n: n.loss.Ds_profile),
        ('Leakage', lambda n: n.loss.Ds_leakage),
        ('Incidence', lambda n: n.loss.Ds_incidence),
        ('Windage', lambda n: n.loss.Ds_secondary),
        ('Mixing', lambda n: n.loss.Ds_mixing),
    ],
}
COMPONENT_NODES = {
    'Nozzle': (0, 1),
    'Interspace': (2, 3),
    'Rotor': (4, 5),
}


def delta_eta_breakdown(data) -> dict[tuple[str, str], float]:
    """
    Total-total efficiency loss shares,
    delta_eta = T_is,out * delta_s / delta_h_tt,is.

    T_is,out is the isentropic exit total temperature and delta_h_tt,is the
    total-to-total isentropic enthalpy drop. Keys are (group, mechanism) with
    group 'Stator', 'Impeller' or 'Exit KE'. Exit KE is the equivalent entropy
    rise from the loss table and is not part of the loss total.
    """
    t_is_out = float(data[n5.oth.Tis_tot][0])
    dh_tt_is = float(data[n0.tot.Enthalpy][0]) - float(data[n5.oth.Enthalpy_totIs][0])
    scale = t_is_out / dh_tt_is

    shares = {}
    for component, mechanisms in LOSS_MECHANISMS.items():
        group = {'Nozzle': 'Stator', 'Rotor': 'Impeller'}[component]
        n_out = NodeVariables(COMPONENT_NODES[component][1])
        for mechanism, spec_fn in mechanisms:
            shares[(group, mechanism)] = scale * float(data[spec_fn(n_out)][0])
    # Interspace entropy rise is grouped with the stator, as in the loss plots
    ds_interspace = float(data[n3.stc.Entropy][0]) - float(data[n2.stc.Entropy][0])
    shares[('Stator', 'Interspace')] = scale * ds_interspace
    ds_exit_ke = float(data[n5.kin.V_mag][0]) ** 2 / (
        2 * float(data[n5.stc.Temperature][0])
    )
    shares[('Exit KE', 'Kinetic energy')] = scale * ds_exit_ke
    return shares


GROUPS = ('Stator', 'Impeller', 'Exit KE')

# Reference station states from the "Present" columns of the comparison table
# (Air, CoolProp). Rotor inlet total equals the nozzle outlet total there.
TABLE_INLET_TOTAL = (580.4e3, 1056.5)  # Pa, K, same as the inlet boundary
TABLE_NOZZLE_OUT_TOTAL = (538.6e3, 1056.5)  # Pa, K
TABLE_ROTOR_OUT_TOTAL = (104.5e3, 731.8)  # Pa, K
TABLE_ROTOR_OUT_STATIC = (94.7e3, 713.9)  # Pa, K
TABLE_ROTOR_OUT_MACH = 0.384


def table_delta_eta_breakdown() -> dict[str, float]:
    """Total-total efficiency loss shares from the reference table, built the same
    way as delta_eta_breakdown: entropy rises from total states, exit KE from
    the outlet Mach number. Returns fractions keyed by group."""

    abs_state = DebugAbstractState('HEOS', 'Air')

    def s_tot(state: tuple[float, float]) -> float:
        abs_state.update(cp.PT_INPUTS, *state)
        return abs_state.smass()

    s_in = s_tot(TABLE_INLET_TOTAL)
    ds_stator = s_tot(TABLE_NOZZLE_OUT_TOTAL) - s_in
    ds_rotor = s_tot(TABLE_ROTOR_OUT_TOTAL) - s_tot(TABLE_NOZZLE_OUT_TOTAL)

    # Isentropic total-to-total outlet state and enthalpy drop
    abs_state.update(cp.PT_INPUTS, *TABLE_INLET_TOTAL)
    h_in = abs_state.hmass()
    abs_state.update(cp.PSmass_INPUTS, TABLE_ROTOR_OUT_TOTAL[0], s_in)
    h_is = abs_state.hmass()
    t_is_out = abs_state.T()
    scale = t_is_out / (h_in - h_is)

    # Exit KE as an equivalent entropy rise, V^2 / (2 T), with V = M * a
    abs_state.update(cp.PT_INPUTS, *TABLE_ROTOR_OUT_STATIC)
    a_out = abs_state.speed_sound()
    ds_exit_ke = (TABLE_ROTOR_OUT_MACH * a_out) ** 2 / (2 * TABLE_ROTOR_OUT_STATIC[1])
    return {
        'Stator': scale * ds_stator,
        'Impeller': scale * ds_rotor,
        'Exit KE': scale * ds_exit_ke,
    }


UA0_SWEEP = np.linspace(0.55, 0.8, 50)  # descending: continue away from design


bc_keys = list(ntw.system.data.boun_cond.keys())
omega_idx = bc_keys.index(n5.kin.Omega)

# a0 is the spouting velocity sqrt(2 * dH_is), with dH_is the total-to-static
# isentropic enthalpy drop (inlet total enthalpy -> isentropic static enthalpy
# at the rotor exit pressure). Set by the boundary conditions, so it is fixed
# along the sweep.
dh_is = float(sol_data[n0.tot.Enthalpy][0]) - float(sol_data[n5.oth.Enthalpy_Is][0])
a0_spout = math.sqrt(2 * dh_is)
omega_design = float(sol_data[n5.kin.Omega][0])
print(
    f'\nDesign point: U/a0 = {omega_design * R_ROTOR_INLET / a0_spout:.3f} '
    f'(a0 = {a0_spout:.1f} m/s, dH_is = {dh_is / 1e3:.1f} kJ/kg, '
    f'omega = {omega_design:.1f} rad/s)'
)

sweep_ua0 = []
sweep_eta_tt = []
# Stator (nozzle) outlet absolute flow angle [deg] and Mach number
sweep_alpha_stator = []
sweep_mach_stator = []
# Total mass flow [kg/s] at each converged sweep point
sweep_mdot = []
# Interspace entropy rise [J/kg/K] at each converged sweep point
sweep_ds_interspace = []
# Entropy rise per loss mechanism at each converged sweep point, per component
sweep_loss = {
    component: {name: [] for name, _ in mechanisms}
    for component, mechanisms in LOSS_MECHANISMS.items()
}
# Total-total efficiency loss shares per sweep point (see delta_eta_breakdown)
sweep_deta = []
# Impeller inlet incidence [deg], beta_in - metal angle at the rotor leading edge
sweep_incidence = []
sol_sweep = sol
if RUN_SPEEDLINE:
    for ua0 in UA0_SWEEP:
        omega_target = ua0 * a0_spout / R_ROTOR_INLET
        kn_sweep = list(kn)
        kn_sweep[omega_idx] = kn[omega_idx] * (omega_target / omega_design)

        try:
            sol_try = solve_root_problem(rtfn_k, sol_sweep, kn_sweep)
        except RuntimeError:
            logger.info(f'U/a0 = {ua0:.3f}: did not converge, skipped')
            continue

        sol_sweep = sol_try
        sweep_data = ntw.system.sol_to_dict(sol_sweep)

        eta_tt = float(sweep_data[n5.ndim.EtaTT][0])
        sweep_ua0.append(ua0)
        sweep_eta_tt.append(eta_tt)
        sweep_alpha_stator.append(
            math.degrees(float(sweep_data[n1.kin.FlowAngleAbs][0]))
        )
        sweep_mach_stator.append(float(sweep_data[n1.kin.Mach][0]))
        sweep_mdot.append(float(sweep_data[n0.oth.TotMassFlow][0]))
        sweep_ds_interspace.append(
            float(sweep_data[n3.stc.Entropy][0]) - float(sweep_data[n2.stc.Entropy][0])
        )
        for component, mechanisms in LOSS_MECHANISMS.items():
            n_out = NodeVariables(COMPONENT_NODES[component][1])
            for mechanism, spec_fn in mechanisms:
                sweep_loss[component][mechanism].append(
                    float(sweep_data[spec_fn(n_out)][0])
                )
        sweep_deta.append(delta_eta_breakdown(sweep_data))
        sweep_incidence.append(
            math.degrees(
                float(sweep_data[n4.kin.FlowAngleRel][0])
                - float(sweep_data[n4.geo.MetalAngle][0])
            )
        )
        logger.info(f'U/a0 = {ua0:.3f}: eta_tt = {eta_tt:.4f}')

    sweep_ua0 = np.array(sweep_ua0)
    sweep_eta_tt = np.array(sweep_eta_tt)
    sweep_alpha_stator = np.array(sweep_alpha_stator)
    sweep_mach_stator = np.array(sweep_mach_stator)
    sweep_mdot = np.array(sweep_mdot)
    sweep_ds_interspace = np.array(sweep_ds_interspace)
    sweep_incidence = np.array(sweep_incidence)
    sweep_deta_group = {
        g: np.array(
            [sum(v for (grp, _), v in s.items() if grp == g) for s in sweep_deta]
        )
        for g in GROUPS
    }

    print('\nU/a0 sweep (converged points only):')
    print(
        tabulate(
            [
                [
                    f'{ua0:.3f}',
                    f'{eta:.4f}',
                    f'{mdot:.4f}',
                    f'{inc:.1f}',
                ]
                for ua0, eta, mdot, inc in zip(
                    sweep_ua0, sweep_eta_tt, sweep_mdot, sweep_incidence
                )
            ],
            headers=['U/a0', 'eta_tt', 'mdot\n[kg/s]', 'incidence\n[deg]'],
            tablefmt='github',
        )
    )

    # Speedline results as CSV (plotting data and the error summary are built from it)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sweep_columns = {
        'ua0': sweep_ua0,
        'eta_tt': sweep_eta_tt,
        'mdot_kg_s': sweep_mdot,
        'incidence_deg': sweep_incidence,
        'alpha1_deg': sweep_alpha_stator,
        'M1': sweep_mach_stator,
        'ds_interspace': sweep_ds_interspace,
    }
    for component, mechanisms in LOSS_MECHANISMS.items():
        for name, _ in mechanisms:
            sweep_columns[f'ds_{component.lower()}_{name.lower()}'] = sweep_loss[
                component
            ][name]
    for group in GROUPS:
        sweep_columns[f'deta_{group.lower().replace(" ", "_")}'] = sweep_deta_group[
            group
        ]
    header = ','.join(sweep_columns)
    rows_csv = np.column_stack(list(sweep_columns.values()))
    np.savetxt(
        DATA_DIR / 't100_speedline.csv',
        rows_csv,
        delimiter=',',
        header=header,
        comments='',
        fmt='%.6g',
    )


# ============================================================
# Pressure-ratio sweep at design shaft speed: PR = p0_in / p5_static. The
# rotor exit static pressure is a fixed boundary condition, so the solved
# network is reused and only that entry of the boundary-condition vector is
# rescaled per point. Each branch is warm-started outwards from the design
# point, so the solver never has to jump across the whole range at once.
# ============================================================
PR_SWEEP = np.linspace(3.0, 9.0, 25)

if RUN_PR_SWEEP:
    p0_in = float(sol_data[n0.tot.Pressure][0])
    p5_design = float(sol_data[n5.stc.Pressure][0])
    pr_design = p0_in / p5_design
    p5_idx = bc_keys.index(n5.stc.Pressure)
    rtfn_ip_pr = ntw.system.make_rootfinder(
        'ipopt',
        opts={'error_on_fail': True, 'ipopt.max_wall_time': 1.0},
    )

    # (PR, U/a0, eta_tt, specific work [kJ/kg]) at each converged sweep point
    pr_rows = []
    pr_branches = (
        [pr for pr in PR_SWEEP if pr >= pr_design],  # ascending from design
        [pr for pr in PR_SWEEP[::-1] if pr < pr_design],  # descending from design
    )
    for branch in pr_branches:
        sol_pr = sol
        for pr in branch:
            kn_pr = list(kn)
            kn_pr[p5_idx] = kn[p5_idx] * (p0_in / (pr * p5_design))

            try:
                sol_try = solve_root_problem(rtfn_k, sol_pr, kn_pr)
            except RuntimeError:
                # Short IPOPT run, then a KINSOL pass to polish its result
                try:
                    sol_ip = solve_root_problem(rtfn_ip_pr, sol_pr, kn_pr, bnd)
                    sol_try = solve_root_problem(rtfn_k, sol_ip, kn_pr)
                except RuntimeError:
                    logger.info(f'PR = {pr:.3f}: did not converge, skipped')
                    continue

            sol_pr = sol_try
            pr_data = ntw.system.sol_to_dict(sol_pr)

            h0_pt = float(pr_data[n0.tot.Enthalpy][0])
            dh_is_pt = h0_pt - float(pr_data[n5.oth.Enthalpy_Is][0])
            ua0_pt = (
                float(pr_data[n5.kin.Omega][0])
                * R_ROTOR_INLET
                / math.sqrt(2 * dh_is_pt)
            )
            eta_tt = float(pr_data[n5.ndim.EtaTT][0])
            work_pt = (h0_pt - float(pr_data[n5.tot.Enthalpy][0])) / 1e3
            pr_rows.append((pr, ua0_pt, eta_tt, work_pt))
            logger.info(f'PR = {pr:.3f}: eta_tt = {eta_tt:.4f}')

    pr_rows.sort()

    print(f'\nPressure-ratio sweep at design speed (design PR = {pr_design:.2f}):')
    print(
        tabulate(
            [
                [f'{pr:.2f}', f'{ua0:.3f}', f'{eta:.4f}', f'{work:.1f}']
                for pr, ua0, eta, work in pr_rows
            ],
            headers=['PR', 'U/a0', 'eta_tt', 'Specific work\n[kJ/kg]'],
            tablefmt='github',
        )
    )


# *************** ************ ***************
# *************** POST-PROCESS ***************
# *************** ************ ***************
active_stations = [
    (NodeVariables(node_idx), node_idx, STATION_LABELS[node_idx])
    for node_idx in range(2 * ntw.num_components)
]


def _val(spec) -> float:
    return float(sol_data[spec][0])


rows = [
    [
        f'{_val(n.tot.Pressure) / 1e5:.3f}',
        f'{_val(n.tot.Temperature):.1f}',
    ]
    for n, _, _ in active_stations
]

print(
    tabulate(
        rows,
        headers=[
            'p0\n[bar]',
            'T0\n[K]',
        ],
        tablefmt='github',
    )
)

# ============================================================
# Loss breakdown and comparison against Jones (1996) Table 2/3
# ============================================================
loss_rows = []
ds_total_all = 0.0
for component, (inlet_idx, outlet_idx) in COMPONENT_NODES.items():
    n_out = NodeVariables(outlet_idx)
    ds_component = _val(n_out.stc.Entropy) - _val(NodeVariables(inlet_idx).stc.Entropy)
    ds_total_all += ds_component

    mechanisms = LOSS_MECHANISMS.get(component, [])
    if not mechanisms:
        loss_rows.append([component, '(isentropic, no loss model)', f'{0.0:.3f}'])
    for mechanism, spec_fn in mechanisms:
        loss_rows.append([component, mechanism, f'{_val(spec_fn(n_out)):.3f}'])

    loss_rows.append([component, 'Subtotal (s_out - s_in)', f'{ds_component:.3f}'])

# Exit kinetic energy leaves the machine rather than being generated as entropy
# inside the rotor, so it is shown as an equivalent entropy rise (exit KE fully
# dissipated at the exit static temperature) and kept out of the subtotal/total
ds_exit_ke = _val(n5.kin.V_mag) ** 2 / (2 * _val(n5.stc.Temperature))
loss_rows.append(['Rotor', 'Exit KE (equiv., not in subtotal)', f'{ds_exit_ke:.3f}'])

loss_rows.append(['All components', 'TOTAL (s5 - s0)', f'{ds_total_all:.3f}'])

print(
    tabulate(
        loss_rows,
        headers=['Component', 'Loss mechanism', 'Ds\n[J/kg/K]'],
        tablefmt='github',
    )
)

# --- Total-total efficiency loss shares at the design point
deta_design = delta_eta_breakdown(sol_data)
deta_rows = [[grp, mech, f'{v * 100:.2f}'] for (grp, mech), v in deta_design.items()]
deta_loss_total = sum(v for (grp, _), v in deta_design.items() if grp != 'Exit KE')
deta_rows.append(['Stator + Impeller', 'Subtotal', f'{deta_loss_total * 100:.2f}'])
deta_rows.append(['', '1 - eta_tt (check)', f'{(1 - _val(n5.ndim.EtaTT)) * 100:.2f}'])
print('\nTotal-total efficiency loss breakdown (design point):')
print(
    tabulate(
        deta_rows,
        headers=['Group', 'Mechanism', 'Delta eta_tt\n[%]'],
        tablefmt='github',
    )
)

# --- Overall performance vs. Sauret Table 3 (engine conditions, "Present"
# 1D analysis) and Jones (1996) Table 2 (design goal)
h0 = _val(n0.tot.Enthalpy)
h5 = _val(n5.tot.Enthalpy)
specific_work = h0 - h5

print(f'\nSpecific work: {specific_work / 1e3:.1f} kJ/kg')
print('Sauret (Table 3): 366 kJ/kg')
print(
    f'Rotor-exit pressure ratio (p0_0 / p5_static): '
    f'{_val(n0.tot.Pressure) / _val(n5.stc.Pressure):.3f} '
    f'(Sauret (Table 4): {580400 / 94700.0:.3f})'
)

# ============================================================
# Loss breakdown bar charts: stator and rotor side by side on one shared
# entropy axis (same scale, so the two panels are directly comparable)
# ============================================================
if PLOTS:
    SURFACE = '#fcfcfb'
    TEXT_PRIMARY = '#0b0b0b'
    TEXT_SECONDARY = '#52514e'
    MUTED = '#898781'
    GRIDLINE = '#e1e0d9'
    BAR_COLOR = '#2a78d6'  # categorical slot 1 (blue); one series per panel

    fig_loss, axs_loss = plt.subplots(
        1, 2, figsize=(10, 4), dpi=70, sharex=True, facecolor=SURFACE
    )
    ds_max = 0.0
    for ax, component in zip(axs_loss, ('Nozzle', 'Rotor')):
        n_out = NodeVariables(COMPONENT_NODES[component][1])
        mechanisms = [
            (name, _val(spec_fn(n_out))) for name, spec_fn in LOSS_MECHANISMS[component]
        ]
        if component == 'Nozzle':
            # The interspace sits between nozzle and rotor and is grouped with the
            # stator side. InterspaceLoss sets the outlet entropy directly, so its
            # rise is the station-to-station entropy difference
            ds_interspace = _val(
                NodeVariables(COMPONENT_NODES['Interspace'][1]).stc.Entropy
            ) - _val(NodeVariables(COMPONENT_NODES['Interspace'][0]).stc.Entropy)
            mechanisms.append(('Interspace', ds_interspace))
        # Sorted ascending so the largest loss sits at the top of the panel
        mechanisms.sort(key=lambda item: item[1])
        names = [name for name, _ in mechanisms]
        values = [value for _, value in mechanisms]
        total = sum(values)
        colors = [BAR_COLOR] * len(values)
        labels = [f'{v:.1f}  ({v / total:.0%})' for v in values]
        if component == 'Rotor':
            # Exit KE leaves the machine rather than being generated in the
            # rotor, so it gets a muted bar at the bottom with no percentage and
            # is left out of the panel total
            names.insert(0, 'Exit KE')
            values.insert(0, ds_exit_ke)
            colors.insert(0, MUTED)
            labels.insert(0, f'{ds_exit_ke:.1f}  (not in total)')
        ds_max = max(ds_max, max(values))

        bars = ax.barh(names, values, height=0.6, color=colors)
        ax.bar_label(
            bars,
            labels=labels,
            padding=4,
            color=TEXT_PRIMARY,
            fontsize=9,
        )
        ax.set_title(
            f'{component}: total {total:.1f} J/kg/K',
            loc='left',
            color=TEXT_PRIMARY,
            fontsize=11,
        )
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color=GRIDLINE, linewidth=1)
        ax.tick_params(colors=TEXT_SECONDARY)
        ax.set_xlabel('Entropy rise Ds [J/kg/K]', color=MUTED)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(MUTED)

    axs_loss[0].set_xlim(0, ds_max * 1.25)
    fig_loss.suptitle(
        'Entropy generation by loss mechanism (T-100 design point)',
        color=TEXT_PRIMARY,
        x=0.02,
        ha='left',
    )
    fig_loss.tight_layout()

# ============================================================
# Total-to-total efficiency against U/a0 (speed sweep)
# ============================================================
if PLOTS and RUN_SPEEDLINE:
    # Experimental eta_T-T from Sauret (2012) Fig. 12 red '+' markers, extracted
    # from the vector drawing data of the PDF (not eyeballed); U/c0 taken as U/a0.
    # Two markers at U/c0 = 0.603 are both drawn (eta 86.08 and 86.11).
    EXP_UA0 = [
        0.603,
        0.603,
        0.629,
        0.643,
        0.646,
        0.648,
        0.654,
        0.655,
        0.679,
        0.705,
        0.727,
        0.749,
        0.760,
        0.767,
        0.775,
        0.780,
        0.791,
    ]
    EXP_ETA_TT = [
        0.8608,
        0.8611,
        0.8558,
        0.8621,
        0.8640,
        0.8801,
        0.8758,
        0.8669,
        0.8730,
        0.8771,
        0.8780,
        0.8821,
        0.8868,
        0.8828,
        0.8818,
        0.8831,
        0.8818,
    ]
    # Sauret (2012) RANS (Axcent-PushButton) total-total efficiency,
    # Fig. 12 blue squares ("CFD with Clearances") and green triangles ("CFD
    # without Clearances"), extracted from the vector drawing data (marker
    # bounding-box centres). Abscissa is U/c0, taken as U/a0 as for the experiments.
    # The model includes the tip gaps, so the "with Clearances" set is the
    # like-for-like comparison.
    CFD_UA0 = [0.564, 0.599, 0.635, 0.670, 0.705, 0.740, 0.776]
    CFD_ETA_TT_CLR = [0.8492, 0.8672, 0.8815, 0.8903, 0.8948, 0.8981, 0.8964]
    CFD_ETA_TT_NOCLR = [0.8575, 0.8781, 0.8905, 0.8990, 0.9054, 0.9066, 0.9052]
    # Efficiency (left) and error (right) side by side in one figure
    fig_sweep, (ax_sweep, ax_err) = plt.subplots(
        1, 2, figsize=(6.5, 3.1), dpi=300, gridspec_kw={'width_ratios': [1.8, 1]}
    )
    ax_sweep.plot(sweep_ua0, sweep_eta_tt, color='#2a78d6', label='ADeT')
    ax_sweep.scatter(
        EXP_UA0,
        EXP_ETA_TT,
        marker='+',
        s=25,
        color='#eb6834',
        label='Experiment (Sauret 2012)',
    )
    ax_sweep.scatter(
        CFD_UA0,
        CFD_ETA_TT_CLR,
        marker='s',
        s=12,
        color='#1baf7a',
        label='CFD with clearances (Sauret 2012)',
    )
    ax_sweep.scatter(
        CFD_UA0,
        CFD_ETA_TT_NOCLR,
        marker='^',
        s=12,
        facecolors='none',
        edgecolors='#1baf7a',
        label='CFD without clearances (Sauret 2012)',
    )
    ax_sweep.legend(frameon=False, loc='lower right', fontsize=7)
    ax_sweep.set_xlabel(r'$U / a_0$ [-]')
    ax_sweep.set_ylabel(r'$\eta_{tt}$ [-]')
    ax_sweep.grid(True, alpha=0.3)

    # --- Error of the model along the speedline against the experiment and the
    # CFD. The model is interpolated at the abscissa of each reference point
    # (sweep_ua0 is sorted ascending for np.interp). Error = model - reference,
    # in percentage points of eta_tt.
    order = np.argsort(sweep_ua0)
    model_at = lambda x: np.interp(np.asarray(x), sweep_ua0[order], sweep_eta_tt[order])
    error_sets = {
        'Experiment (Sauret 2012)': (
            np.asarray(EXP_UA0),
            100 * (model_at(EXP_UA0) - np.asarray(EXP_ETA_TT)),
            '#eb6834',
            '+',
        ),
        'CFD with clearances': (
            np.asarray(CFD_UA0),
            100 * (model_at(CFD_UA0) - np.asarray(CFD_ETA_TT_CLR)),
            '#1baf7a',
            's',
        ),
        'CFD without clearances': (
            np.asarray(CFD_UA0),
            100 * (model_at(CFD_UA0) - np.asarray(CFD_ETA_TT_NOCLR)),
            '#1baf7a',
            '^',
        ),
    }
    print('\nSpeedline error, model - reference [eta points]:')
    error_summary_rows = []
    for name, (_, err, _, _) in error_sets.items():
        error_summary_rows.append(
            [
                name,
                f'{err.mean():+.2f}',
                f'{np.sqrt(np.mean(err**2)):.2f}',
                f'{np.abs(err).max():.2f}',
            ]
        )
    print(
        tabulate(
            error_summary_rows,
            headers=['Reference', 'Mean error', 'RMS error', 'Max |error|'],
            tablefmt='github',
        )
    )

    for name, (x_ref, err, color, marker) in error_sets.items():
        ax_err.scatter(
            x_ref,
            err,
            marker=marker,
            s=20,
            color=color,
            facecolors='none' if marker == '^' else color,
            label=name,
        )
    ax_err.axhline(0, color='gray', linewidth=0.8)
    ax_err.set_xlabel(r'$U / a_0$ [-]')
    ax_err.set_ylabel(r'$\eta_{tt,\mathrm{ADeT}} - \eta_{tt,\mathrm{ref}}$ [\%-pts]')
    ax_err.grid(True, alpha=0.3)
    # Headroom above the data so the legend sits clear of the points
    all_err = np.concatenate([err for _, err, _, _ in error_sets.values()])
    err_lo = min(all_err.min(), 0) - 0.5
    err_hi = max(all_err.max(), 0) + 0.3
    ax_err.set_ylim(err_lo, err_hi + 0.6 * (err_hi - err_lo))
    ax_err.legend(frameon=False, loc='upper right', fontsize=6.5)
    fig_sweep.tight_layout()
    if SAVE_FIGURES:
        save_pdf(fig_sweep, FIG_DIR / 'adet_t100_speedline.pdf')

# ============================================================
# Impeller inlet incidence against total-to-total efficiency along the speedline
# ============================================================
if PLOTS and RUN_SPEEDLINE:
    # Model only: the reference (Jones 1996) trend is overlaid separately
    fig_inc, ax_inc = plt.subplots(figsize=(3.6, 2.5), dpi=300)
    ax_inc.plot(
        sweep_incidence, sweep_eta_tt, marker='o', ms=3, color='#2a78d6', label='ADeT'
    )
    # Label every fifth point with its U/a0 so the dense part of the curve stays legible
    for ua0, inc, eta in list(zip(sweep_ua0, sweep_incidence, sweep_eta_tt))[::5]:
        ax_inc.annotate(
            f'{ua0:.2f}',
            (inc, eta),
            textcoords='offset points',
            xytext=(4, 4),
            fontsize=6,
            color=TEXT_SECONDARY,
        )
    ax_inc.axvline(0.0, color='gray', linestyle='--', linewidth=0.8)
    ax_inc.set_ylim(0.80, 0.90)
    ax_inc.set_xlabel(r'Impeller inlet incidence $i$ [deg]')
    ax_inc.set_ylabel(r'$\eta_{tt}$ [-]')
    ax_inc.grid(True, alpha=0.3)
    ax_inc.legend(frameon=False, loc='lower right')
    fig_inc.tight_layout()
    if SAVE_FIGURES:
        save_pdf(fig_inc, FIG_DIR / 'adet_t100_eta_incidence.pdf')

# ============================================================
# Total-to-total efficiency against pressure ratio (design speed)
# ============================================================
if PLOTS and RUN_PR_SWEEP and pr_rows:
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5), dpi=70)
    pr_axis = [row[0] for row in pr_rows]
    pr_eta = [row[2] for row in pr_rows]
    ax_pr.plot(pr_axis, pr_eta, marker='o', color='#2a78d6')
    ax_pr.axvline(
        pr_design, color='gray', linestyle='--', label=f'Design PR = {pr_design:.2f}'
    )
    ax_pr.set_xlabel(r'$p_{0,in} / p_{5,s}$ [-]')
    ax_pr.set_ylabel(r'$\eta_{tt}$ [-]')
    ax_pr.set_title('Total-to-total efficiency vs. pressure ratio (T-100)')
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(frameon=False)
    fig_pr.tight_layout()

# ============================================================
# Loss breakdown along the speedline: stacked delta_eta_tt [%] per mechanism vs.
# U/a0, stator and impeller in separate panels on a shared scale. Same shares
# as the design-point bars (delta_eta_breakdown), so the two plots agree.
# ============================================================
if PLOTS and RUN_SPEEDLINE:
    # Colour follows the mechanism, not its position in the stack, so a given
    # loss keeps its hue in both panels (categorical slots 1-5, in order)
    MECHANISM_COLORS = {
        'Profile': '#2a78d6',
        'Leakage': '#eb6834',
        'Incidence': '#1baf7a',
        'Windage': '#8c564b',
        'Endwall': '#eda100',
        'Mixing': '#e87ba4',
        'Interspace': '#7d5ba6',
    }

    fig_stack, axs_stack = plt.subplots(
        1, 2, figsize=(6.5, 2.9), dpi=300, sharex=True, sharey=True, facecolor=SURFACE
    )
    # Group keys of delta_eta_breakdown: ('Stator', ...) for the nozzle and the
    # interspace, ('Impeller', ...) for the rotor
    for ax, (component, group) in zip(axs_stack, (('Nozzle', 'Stator'), ('Rotor', 'Impeller'))):
        names = [name for name, _ in LOSS_MECHANISMS[component]]
        if component == 'Nozzle':
            # The interspace is grouped with the stator side, as in the design-point plots
            names.append('Interspace')
        series = [
            100 * np.array([s[(group, name)] for s in sweep_deta]) for name in names
        ]
        ax.stackplot(
            sweep_ua0,
            *series,
            labels=names,
            colors=[MECHANISM_COLORS[name] for name in names],
            edgecolor=SURFACE,
            linewidth=0.8,
        )
        ax.set_title(component, loc='left', color=TEXT_PRIMARY, fontsize=11)
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRIDLINE, linewidth=1)
        ax.tick_params(colors=TEXT_SECONDARY)
        ax.set_xlabel(r'$U / a_0$ [-]', color=MUTED)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.spines['left'].set_color(MUTED)
        ax.spines['bottom'].set_color(MUTED)
        ax.legend(loc='best', frameon=False, fontsize=7)

    axs_stack[0].set_ylabel(r'$\Delta\eta_{tt}$ [\%]', color=MUTED)
    fig_stack.tight_layout()
    if SAVE_FIGURES:
        save_pdf(fig_stack, FIG_DIR / 'adet_t100_loss_stack.pdf')

# ============================================================
# Total-to-total efficiency loss at the design point: solver against the
# reference table, stator, impeller and exit KE as grouped bars, delta_eta_tt [%]
# (exit KE sits outside the loss total)
# ============================================================
if PLOTS:
    SERIES_COLORS = {'Solver': '#2a78d6', 'Table (Present)': '#eb6834'}
    GROUP_LABELS = {
        'Stator': 'Stator',
        'Impeller': 'Impeller',
        'Exit KE': 'Exit KE',
    }
    deta_table = table_delta_eta_breakdown()
    deta_series = {
        'Solver': [
            100 * sum(v for (grp, _), v in deta_design.items() if grp == g)
            for g in GROUPS
        ],
        'Table (Present)': [100 * deta_table[g] for g in GROUPS],
    }

    fig_eta, ax_eta = plt.subplots(figsize=(5.2, 3.2), dpi=300, facecolor=SURFACE)
    x_groups = np.arange(len(GROUPS))
    bar_width = 0.38
    for i, (label, values) in enumerate(deta_series.items()):
        bars_eta = ax_eta.bar(
            x_groups + (i - 0.5) * bar_width,
            values,
            width=bar_width,
            color=SERIES_COLORS[label],
            label=label,
        )
        ax_eta.bar_label(
            bars_eta,
            labels=[rf'{v:.2f}\%' for v in values],
            padding=3,
            color=TEXT_PRIMARY,
            fontsize=8,
        )
    ax_eta.set_xticks(x_groups, [GROUP_LABELS[g] for g in GROUPS])

    ax_eta.set_facecolor(SURFACE)
    ax_eta.set_axisbelow(True)
    peak_eta = max(max(values) for values in deta_series.values())
    ax_eta.set_ylim(0, peak_eta * 1.2)
    ax_eta.grid(axis='y', color=GRIDLINE, linewidth=0.6)
    ax_eta.tick_params(colors=TEXT_SECONDARY)
    ax_eta.set_ylabel(r'$\Delta\eta_{tt}$ [\%]', color=MUTED)
    ax_eta.legend(frameon=False, loc='upper right')
    for side in ('top', 'right'):
        ax_eta.spines[side].set_visible(False)
    ax_eta.spines['left'].set_color(MUTED)
    ax_eta.spines['bottom'].set_color(MUTED)
    fig_eta.tight_layout()
    if SAVE_FIGURES:
        save_pdf(fig_eta, FIG_DIR / 'adet_t100_deta_components.pdf')

# ============================================================
# Mach numbers at the reference stations against Sauret (2012) Table 4
# ("Present" RITAL column): absolute (model - paper) and relative difference.
# M is the absolute Mach number, Mw the relative one. Note the rotor outlet
# relative Mach is a boundary condition (set to Sauret's value), so its
# difference is zero by construction.
# ============================================================
if PLOTS:
    MACH_STATIONS = [
        # (label, Sauret Table 4 "Present" value, model value at the solution)
        ('Nozzle outlet\nM', 0.887, _val(n1.kin.Mach)),
        ('Rotor inlet\nM', 0.986, _val(n4.kin.Mach)),
        ('Rotor inlet\nMw', 0.265, _val(n4.kin.RelMach)),
        ('Rotor outlet\nM', 0.384, _val(n5.kin.Mach)),
        ('Rotor outlet\nMw', 0.713, _val(n5.kin.RelMach)),
    ]
    mach_labels = [label for label, _, _ in MACH_STATIONS]
    mach_paper = np.array([paper for _, paper, _ in MACH_STATIONS])
    mach_model = np.array([model for _, _, model in MACH_STATIONS])
    mach_abs_diff = mach_model - mach_paper
    mach_rel_diff = 100 * mach_abs_diff / mach_paper

    fig_mach, axs_mach = plt.subplots(
        1, 2, figsize=(14, 6), dpi=70, sharex=True, facecolor=SURFACE
    )
    x_mach = np.arange(len(MACH_STATIONS))
    mach_panels = [
        (
            axs_mach[0],
            mach_abs_diff,
            [f'{v:+.3f}' for v in mach_abs_diff],
            r'$\Delta M$ [-]',
            'Absolute: model - paper',
        ),
        (
            axs_mach[1],
            mach_rel_diff,
            [f'{v:+.1f}%' for v in mach_rel_diff],
            r'$\Delta M / M_{paper}$ [%]',
            'Relative: (model - paper) / paper',
        ),
    ]
    for ax, values, bar_text, ylabel, title in mach_panels:
        bars_mach = ax.bar(x_mach, values, width=0.6, color=BAR_COLOR)
        ax.bar_label(
            bars_mach,
            labels=bar_text,
            padding=4,
            color=TEXT_PRIMARY,
            fontsize=12,
        )
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(x_mach, mach_labels, fontsize=12)
        ax.set_title(title, loc='left', color=TEXT_PRIMARY, fontsize=13)
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRIDLINE, linewidth=1)
        ax.tick_params(colors=TEXT_SECONDARY)
        ax.set_ylabel(ylabel, color=MUTED)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.spines['left'].set_color(MUTED)
        ax.spines['bottom'].set_color(MUTED)
        lo, hi = min(0.0, values.min()), max(0.0, values.max())
        span = hi - lo
        ax.set_ylim(lo - 0.15 * span, hi + 0.25 * span)
    fig_mach.suptitle(
        'Mach number vs. Sauret (2012) Table 4 "Present" (T-100)',
        color=TEXT_PRIMARY,
        x=0.02,
        ha='left',
        fontsize=14,
    )
    fig_mach.tight_layout()

# ============================================================
# Stator outlet absolute flow angle and Mach number along the speedline, one
# panel per quantity (separate y-scales) sharing the U/a0 axis
# ============================================================
if PLOTS and RUN_SPEEDLINE:
    fig_stator, axs_stator = plt.subplots(
        2, 1, figsize=(7, 7), dpi=70, sharex=True, facecolor=SURFACE
    )
    ax_alpha, ax_mach = axs_stator

    ax_alpha.plot(sweep_ua0, sweep_alpha_stator, marker='o', color='#2a78d6')
    ax_alpha.set_title(
        'Stator outlet absolute flow angle', loc='left', color=TEXT_PRIMARY, fontsize=11
    )
    ax_alpha.set_ylabel(r'$\alpha_1$ [deg]', color=MUTED)

    ax_mach.plot(sweep_ua0, sweep_mach_stator, marker='o', color='#eb6834')
    ax_mach.set_title(
        'Stator outlet Mach number', loc='left', color=TEXT_PRIMARY, fontsize=11
    )
    ax_mach.set_ylabel(r'$M_1$ [-]', color=MUTED)
    ax_mach.set_xlabel(r'$U / a_0$ [-]', color=MUTED)

    for ax in axs_stator:
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(True, color=GRIDLINE, linewidth=1)
        ax.tick_params(colors=TEXT_SECONDARY)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.spines['left'].set_color(MUTED)
        ax.spines['bottom'].set_color(MUTED)

    fig_stator.suptitle(
        'Stator outlet conditions along the speedline (T-100)',
        color=TEXT_PRIMARY,
        x=0.02,
        ha='left',
    )
    fig_stator.tight_layout()

# ============================================================
# Total mass flow along the speedline (U/a0 sweep)
# ============================================================
if PLOTS and RUN_SPEEDLINE:
    fig_mdot, ax_mdot = plt.subplots(figsize=(7, 5), dpi=70, facecolor=SURFACE)
    ax_mdot.plot(sweep_ua0, sweep_mdot, marker='o', color='#2a78d6')
    ax_mdot.set_xlabel(r'$U / a_0$ [-]')
    ax_mdot.set_ylabel(r'$\dot{m}$ [kg/s]')
    ax_mdot.set_title('Mass flow along the speedline (T-100)')
    ax_mdot.grid(True, alpha=0.3)
    fig_mdot.tight_layout()

# ============================================================
# Meridional profile of each component (hub/tip contours + LE/TE), laid
# end to end along the axis: each component starts where the previous one's
# axial chord ends (the interspace has zero chord, so it adds no length)
# ============================================================
COMPONENT_COLORS = {
    'Nozzle': 'darkblue',
    'Interspace': 'gray',
    'Rotor': 'darkred',
}

if PLOTS:
    fig_mer, ax_mer = plt.subplots(figsize=(8, 6), dpi=70)

    axial_offset = 0.0
    for component, (inlet_idx, outlet_idx) in COMPONENT_NODES.items():
        n_in = NodeVariables(inlet_idx)
        n_out = NodeVariables(outlet_idx)

        geom = RowGeometry(
            r_in=_val(n_in.geo.Rmid),
            r_out=_val(n_out.geo.Rmid),
            height_in=_val(n_in.geo.Height),
            height_out=_val(n_out.geo.Height),
            mer_angle_in=_val(n_in.geo.MeridionalAngle),
            mer_angle_out=_val(n_out.geo.MeridionalAngle),
            axial_chord=_val(n_out.geo.MerChord),
            axial_offset=axial_offset,
            force_straight=False,
        )
        lines = geom.plot_meridional_profile(
            color=COMPONENT_COLORS[component], ax=ax_mer
        )
        lines[0].set_label(component)

        axial_offset += geom.axial_chord

    ax_mer.set_aspect('equal')
    ax_mer.set_title('Meridional profile')
    ax_mer.set_xlabel(r'$z$ / [m]')
    ax_mer.set_ylabel(r'$r$ / [m]')
    ax_mer.grid(True, alpha=0.3)
    ax_mer.legend()
    fig_mer.tight_layout()

    # ============================================================
    # Velocity triangle plots at each station
    # ============================================================
    ncols = min(len(active_stations), 3)
    nrows = math.ceil(len(active_stations) / ncols)
    fig, axs_tri = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), dpi=70)
    axs_flat = np.atleast_1d(axs_tri).flatten()

    for ax, (n, node_idx, label) in zip(axs_flat, active_stations):
        ax.set_aspect('equal')
        plot_velocity_triangles(
            sol_data[n.kin.V_tan],
            sol_data[n.kin.V_mer],
            sol_data[n.kin.BladeSpeed],
            sol_data[n.geo.RDistr],
            ax,
        )
        ax.set_title(f'{node_idx}: {label}')

    for ax in axs_flat[len(active_stations) :]:
        ax.set_visible(False)

    fig.tight_layout()
    if not SAVE_FIGURES:
        plt.show(block=True)
