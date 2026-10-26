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
from adet.equations.nondimensional import (
    GammaPV,
    TotalStaticExpansionEfficiency,
    TotalTotalExpansionEfficiency,
)
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.losses.mixing import DentonBaumgartnerMixingLoss, SieverdingBasePressure
from adet.losses.rit import (
    ChenSlip,
    EndwallLoss,
    ImpellerIncidenceLoss,
    ImpellerLeakageLoss,
    ImpellerPassageLoss,
    StatorProfileLoss,
)
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_velocity_triangles
from adet.variables import KinematicVariables, NodeVariables
from adet.varspec import DEF_NODE

logger = logging.getLogger(__name__)
setup_logger(logger)
PLOTS = True
USE_SLIP_MODEL = True


_n0 = NodeVariables(0)
_n1 = NodeVariables(1)


class AddImpellerLosses(LossApplier):
    """Apply the rotor's passage + leakage + incidence + endwall + mixing
    losses to the static entropy rise."""

    def residual(
        self,
        s0: _n0.stc.Entropy.Hint,
        s1: _n1.stc.Entropy.Hint,
        ds_profile1: _n1.loss.Ds_profile.Hint,
        ds_leakage1: _n1.loss.Ds_leakage.Hint,
        ds_incidence1: _n1.loss.Ds_incidence.Hint,
        ds_endwall1: _n1.loss.Ds_endwall.Hint,
        ds_mixing1: _n1.loss.Ds_mixing.Hint,
    ):
        return s1 - (
            s0 + ds_profile1 + ds_leakage1 + ds_incidence1 + ds_endwall1 + ds_mixing1
        )


class AddStatorLosses(LossApplier):
    """Apply the nozzle's profile + endwall + mixing losses to the static
    entropy rise."""

    def residual(
        self,
        s0: _n0.stc.Entropy.Hint,
        s1: _n1.stc.Entropy.Hint,
        ds_profile1: _n1.loss.Ds_profile.Hint,
        ds_endwall1: _n1.loss.Ds_endwall.Hint,
        ds_mixing1: _n1.loss.Ds_mixing.Hint,
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

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)
n4 = NodeVariables(4)
n5 = NodeVariables(5)

inl = Inlet(
    {
        n0.tot.Pressure: Quantity(580400, 'Pa'),  # Sauret Table 1
        n0.tot.Temperature: Quantity(1056.5, 'K'),  # Sauret Table 1
        n0.oth.TotMassFlow: 0.33,  # kg/s, Sauret Table 1
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
    stator_extra_equations: dict = {ZeroDeviation(): 0}  # No incidence
    rotor_extra_equations: dict = {}
    if USE_SLIP_MODEL:
        rotor_extra_equations[ChenSlip()] = 0

    if include_losses:
        stator_extra_equations.update(
            {
                AddStatorLosses(): (0, 1),
                StatorProfileLoss(): (0, 1),
                EndwallLoss(0.0025): (0, 1),
                IsentropicProperties(): (0, 1),
                BoundaryLayerRatios(): 1,
                GammaPV(): 0,
                GammaPV(): 1,
                SieverdingBasePressure(): (0, 1),
                DentonBaumgartnerMixingLoss(): 1,
            }
        )
        rotor_extra_equations.update(
            {
                AddImpellerLosses(): (0, 1),
                IsentropicProperties(): (0, 1),
                ImpellerPassageLoss(): (0, 1),
                ImpellerLeakageLoss(): (0, 1),
                ImpellerIncidenceLoss(): (0, 1),
                EndwallLoss(): (0, 1),
                BoundaryLayerRatios(): 1,
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
            n1.kin.FlowAngleAbs: Quantity(78.1, 'deg'),  # Sauret Fig. 8(a)
            n0.geo.BldThick: Quantity(0.000661, 'm'),  # Sauret Table 6 (m=0%)
            n1.geo.BldThick: Quantity(0.00051, 'm'),  # Sauret Table 2
            n1.geo.NumBlades: 19,  # Sauret Table 1
            n1.geo.MerChord: Quantity(0.01, 'mm'),
            n1.oth.MomByBld: 0.1,
            n1.oth.DispByMom: 2.0,
            n1.oth.DispByHgt: 0.15,
        },
        shaft=casing,
        extra_equations=stator_extra_equations,
    )

    interspace = Interspace(
        'intrspc',
        {
            # vaneless space outlet (Sauret Table 2)
            n1.geo.Rmid: Quantity(0.0582, 'm'),  # rotor inlet tip radius
            n1.geo.Height: Quantity(0.00635, 'm'),  # vaneless space height
        },
        extra_equations={
            IsentropicLink(): (0, 1),
        },
    )

    rotor = BladeRow(
        'impeller',
        bound_cond={
            n1.geo.Rhub: Quantity(0.01524, 'm'),  # exducer hub radius (Sauret Table 8)
            n1.geo.Height: Quantity(0.03683 - 0.01524, 'm'),  # exducer tip - hub radius
            n1.geo.MeridionalAngle: Quantity(0, 'deg'),  # axial exducer exit
            n0.geo.BldThick: Quantity(0.00086, 'm'),  # Sauret Table 2 / Appendix
            n0.geo.MetalAngle: Quantity(0, 'deg'),  # purely radial LE blades
            n1.geo.BldThick: Quantity(0.00076, 'm'),  # Sauret Table 2 / Appendix
            n1.geo.MerChord: Quantity(0.0389, 'm'),  # axial length, Sauret Table 2
            n1.geo.NumBlades: 16,  # Sauret Table 1 / Jones text
            n1.stc.Pressure: Quantity(94700.0, 'Pa'),  # Sauret Table 4
            n0.geo.TipClearance: Quantity(0.0004, 'm'),  # Sauret Table 2
            n1.geo.TipClearance: Quantity(0.00023, 'm'),  # Sauret Table 2
            n1.oth.MomByBld: 0.075,
            n1.oth.DispByMom: 2.0,
            n1.oth.DispByHgt: 0.09,
        },
        shaft=shaft,
        extra_equations=rotor_extra_equations,
    )

    if USE_SLIP_MODEL:
        # ``ChenSlip`` sets the rotor inlet tangential velocity (hence its
        # relative flow angle) directly
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
    n0.stc.Temperature.Glob: (300, 1300),
    n0.stc.Pressure.Glob: (1e3, 2e6),
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

sol_isen = solve_root_problem(rtfn_isen, x0_isen, kn_isen, suppress_output=False)
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
rtfn = ntw.system.make_rootfinder('kinsol')
sol = solve_root_problem(rtfn, sol, kn, bnd)

sol_data = ntw.system.sol_to_dict(sol)

# ============================================================
# Active machine stations: only the components actually chained
# into the network above have solved nodes (see STATION_LABELS)
# ============================================================
active_stations = [
    (NodeVariables(node_idx), node_idx, STATION_LABELS[node_idx])
    for node_idx in range(2 * ntw.num_components)
]


def _val(spec) -> float:
    return float(sol_data[spec][0])


# *************** ************ ***************
# *************** POST-PROCESS ***************
# *************** ************ ***************

rows = [
    [
        f'{node_idx}: {label}',
        f'{_val(n.stc.Pressure) / 1e5:.3f}',
        f'{_val(n.tot.Pressure) / 1e5:.3f}',
        f'{_val(n.rlt.Pressure) / 1e5:.3f}',
        f'{_val(n.stc.Temperature):.1f}',
        f'{_val(n.tot.Temperature):.1f}',
        f'{_val(n.stc.Density):.3f}',
        f'{_val(n.stc.Entropy):.1f}',
        f'{_val(n.kin.V_mag):.1f}',
        f'{_val(n.kin.W_mag):.1f}',
        f'{_val(n.kin.BladeSpeed):.1f}',
        f'{math.degrees(_val(n.kin.FlowAngleAbs)):.1f}',
        f'{math.degrees(_val(n.kin.FlowAngleRel)):.1f}',
        f'{_val(n.kin.Mach):.3f}',
        f'{_val(n.kin.RelMach):.3f}',
    ]
    for n, node_idx, label in active_stations
]

print(
    tabulate(
        rows,
        headers=[
            'Station',
            'p\n[bar]',
            'p0\n[bar]',
            'p0_rel\n[bar]',
            'T\n[K]',
            'T0\n[K]',
            'rho\n[kg/m3]',
            's\n[J/kg/K]',
            'V\n[m/s]',
            'W\n[m/s]',
            'U\n[m/s]',
            'alpha\n[deg]',
            'beta\n[deg]',
            'M',
            'M_rel',
        ],
        tablefmt='github',
    )
)

# ============================================================
# Loss breakdown and comparison against Jones (1996) Table 2/3
# ============================================================
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
        ('Endwall', lambda n: n.loss.Ds_endwall),
        ('Mixing', lambda n: n.loss.Ds_mixing),
    ],
}
COMPONENT_NODES = {
    'Nozzle': (0, 1),
    'Interspace': (2, 3),
    'Rotor': (4, 5),
}

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
        # Sorted ascending so the largest loss sits at the top of the panel
        mechanisms = sorted(
            [
                (name, _val(spec_fn(n_out)))
                for name, spec_fn in LOSS_MECHANISMS[component]
            ],
            key=lambda item: item[1],
        )
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
# Meridional profile of each component (hub/tip contours + LE/TE), laid
# end to end along the axis: each component starts where the previous one's
# axial chord ends (the interspace has zero chord, so it adds no length)
# ============================================================
COMPONENT_COLORS = {'Nozzle': 'darkblue', 'Interspace': 'gray', 'Rotor': 'darkred'}

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
    plt.show()
