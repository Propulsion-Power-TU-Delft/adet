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
- Missing stator-rotor row gap loss (currently purely isentropic, free-vortex flow)
- Volute not integrated
"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from adet.equations.geometrical import AxialLengthToRadiusChange, MeridionalChord
from pint import Quantity
from tabulate import tabulate

from adet.assemblers import CasadiSystem
from adet.components.blade_row import BladeRow, Interspace
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.base_equation import LossApplier
from adet.equations.definitions import (
    BoundaryLayerRatios,
    EulerWork,
    IsentropicEulerWork,
    IsentropicProperties,
)
from adet.equations.nondimensional import (
    EfficiencyDeficit,
    GammaPV,
    LossCoefficient,
    TotalStaticExpansionEfficiency,
    TotalTotalExpansionEfficiency,
)
from adet.fluid.settings import FluidSettings
from adet.losses.base_loss import collect_loss_mechanisms
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.losses.mixing import DentonBaumgartnerMixingLoss, SieverdingBasePressure
from adet.losses.rit import (
    StatorProfileLoss,
    ChenSlip,
    EndwallLoss,
    ImpellerIncidenceLoss,
    ImpellerLeakageLoss,
    ImpellerPassageLoss,
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


# |> Machine stations (for terminal output / plotting)
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

# ============================================================
# Geometry and boundary conditions. Radii/heights/lengths/angles are
# Sauret Table 2 and Appendix A (Tables 6-11) unless noted otherwise;
# cycle boundary conditions are Sauret Table 1. Set directly as component
# constraints below -- ratios/differences are computed inline from the
# same Sauret source values rather than through standalone variables.
# ============================================================

inl = Inlet(
    {
        # Cycle boundary conditions at the engine design point (Sauret Table 1)
        n0.tot.Pressure: Quantity(580400, 'Pa'),
        n0.tot.Temperature: Quantity(1056.5, 'K'),
        n0.oth.TotMassFlow: Quantity(0.33, 'kg/s').magnitude,
        # NOTE (assumed / Jones-only): Sauret doesn't tabulate nozzle inlet
        # swirl; Jones gives a 20-40 deg range, midpoint used here.
        n0.kin.FlowAngleRel: Quantity(30.0, 'deg'),
        # Nozzle inlet height [m] (Sauret Table 2)
        n0.geo.Height: Quantity(0.00635, 'm'),
        n0.geo.MeridionalAngle: Quantity(-90, 'deg'),
    }
)

casing = Shaft(Quantity(0, 'rpm'), is_constrained=True)
shaft = Shaft(Quantity(106588, 'rpm'), is_constrained=True)


def build_network(include_losses: bool) -> ComponentNetwork:
    """Build the nozzle/interspace/rotor network.

    With ``include_losses=False`` every entropy-generating loss mechanism is
    replaced by an ``IsentropicLink`` (s_out = s_in), giving a much
    easier-to-solve system whose solution is used as the initial guess for
    the full, lossy system (see below).
    """
    stator_extra_equations: dict = {
        ZeroDeviation(): 0,
        MeridionalChord(): (0, 1),
    }  # No incidence
    rotor_extra_equations: dict = {
        MeridionalChord(): (0, 1),
        AxialLengthToRadiusChange(): (0, 1),
    }
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
            # Nozzle heights [m] (Sauret Table 2): 0.006 / 0.00635
            n1.geo.HeightRatio: 0.006 / 0.00635,
            # Nozzle radii [m] (Sauret Table 2, "Nozzle"): 0.0635 / 0.074
            # (n1(exit)/n0(inlet))
            n1.geo.RadiusRatio: 0.0635 / 0.074,
            n1.geo.Rmid: Quantity(0.0635, 'm'),
            n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
            # Nozzle exit angle (Sauret Fig. 8(a), Present/RITAL)
            n1.geo.MetalAngle: Quantity(78.1, 'deg'),
            # Nozzle blade geometry (Sauret Table 2 "Nozzle" + Appendix
            # Table 6, m=0%)
            n0.geo.BldThick: Quantity(0.000661, 'm'),
            n1.geo.BldThick: Quantity(0.00051, 'm'),
            n1.geo.NumBlades: 19,  # Sauret Table 1
            # Planar stator: zero axial length, meridional chord reduces
            # to the inlet/outlet radius change (Table 2)
            n1.geo.AxialLength: Quantity(0.0, 'm'),
            n1.oth.MomByBld: 0.075,
            n1.oth.DispByMom: 2.0,
            n1.oth.DispByHgt: 0.09,
        },
        shaft=casing,
        extra_equations=stator_extra_equations,
    )

    interspace = Interspace(
        'intrspc',
        {
            # Vaneless-space (nozzle exit -> rotor inlet) radius/height
            # ratios (Sauret Table 2): rotor inlet R (0.0582) / nozzle
            # exit R (0.0635), and 0.00635 / nozzle exit height (0.006)
            n1.geo.RadiusRatio: 0.0582 / 0.0635,
            n1.geo.HeightRatio: 0.00635 / 0.006,
        },
        extra_equations={
            IsentropicLink(): (0, 1),
        },
    )

    rotor = BladeRow(
        'impeller',
        bound_cond={
            # Exducer (rotor exit) hub/tip radii [m] (Sauret Table 8, m=100%)
            n1.geo.Rmid: Quantity(0.02604, 'm'),
            n1.geo.Height: Quantity(0.03683 - 0.01524, 'm'),
            n1.geo.MeridionalAngle: Quantity(0, 'deg'),  # axial exducer exit
            # Rotor blade geometry (Sauret Table 2 "Rotor" + Appendix
            # Tables 9-10)
            n0.geo.BldThick: Quantity(0.00086, 'm'),
            n0.geo.MetalAngle: Quantity(0, 'deg'),  # purely radial LE blades
            n1.geo.BldThick: Quantity(0.00076, 'm'),
            n1.ndim.AxLenToRadChangeRatio: 0.67,
            n1.geo.NumBlades: 16,  # Sauret Table 1 / Jones text
            # Rotor exit static pressure: Sauret Table 4, "Present"
            # (RITAL) column, engine conditions
            n1.stc.Pressure: Quantity(94700.0, 'Pa'),
            n0.geo.TipClearance: Quantity(0.0004, 'm'),
            n1.geo.TipClearance: Quantity(0.00023, 'm'),
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

    return ComponentNetwork(
        fluid_settings,
        inl,
        CasadiSystem(1),
        [
            stator,
            interspace,
            rotor,
        ],
    )


# ============================================================
# Bounds
# ============================================================

BOUNDS = {
    n1.kin.Mach: (0.5, 1.05),
    n0.stc.Temperature.Glob: (300, 1300),
    n0.stc.Pressure.Glob: (1e3, 2e6),
    # n1.stc.Pressure: (0.3e5, 8e5),
    # n0.stc.Entropy.Glob: (3000.0, 7000.0),
    n5.kin.FlowAngleAbs: (
        Quantity(-20, 'deg').to('rad').magnitude,
        Quantity(20, 'deg').to('rad').magnitude,
    ),
    n4.kin.FlowAngleRel: (
        Quantity(-89, 'deg').to('rad').magnitude,
        Quantity(0, 'deg').to('rad').magnitude,
    ),
}

# ============================================================
# Part 1: isentropic initialization (no loss models)
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
# Part 2: real system (with loss models)
# ============================================================
ntw = build_network(include_losses=True)
ntw.system.add_equation(TotalStaticExpansionEfficiency(), (0, 5))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 5))
ntw.system.add_equation(EulerWork(), (0, 5))
ntw.system.add_equation(IsentropicEulerWork(), (0, 5))

COMPONENT_NODES = {
    'Nozzle': (0, 1),
    'Interspace': (2, 3),
    'Rotor': (4, 5),
}
MACHINE_EXIT_NODE = 5

component_losses: dict[str, tuple[int, LossCoefficient, EfficiencyDeficit]] = {}
for component, (_, outlet_idx) in COMPONENT_NODES.items():
    mechanisms = collect_loss_mechanisms(ntw.system, outlet_idx)
    loss_coeff = LossCoefficient(mechanisms)
    eff_deficit = EfficiencyDeficit(mechanisms)
    ntw.system.add_equation(loss_coeff, outlet_idx)
    # EfficiencyDeficit drops the machine-reference node entirely when
    # there are no mechanisms (nothing to weigh work_is by), so size the
    # position tuple from its actual arg_nodes rather than assuming 2.
    eff_deficit_position = tuple(
        outlet_idx if local_node == 0 else MACHINE_EXIT_NODE
        for local_node in eff_deficit.arg_nodes
    )
    ntw.system.add_equation(eff_deficit, eff_deficit_position)
    component_losses[component] = (outlet_idx, loss_coeff, eff_deficit)

ntw.build()

rtfn = ntw.system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False},
)
x0 = ntw.system.get_guess({**isen_data, **isen_property_guess}, fallback=0.5)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds(BOUNDS)

sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=True)

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


# ============================================================
# Terminal output: thermodynamic and flow variables at each station
# ============================================================
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
# Loss breakdown: Denton-style loss coefficient (zeta) and the
# resulting efficiency deficit, per mechanism
# ============================================================
loss_rows = []
for component, (outlet_idx, loss_coeff, eff_deficit) in component_losses.items():
    if not loss_coeff.ds_specs:
        loss_rows.append(
            [component, '(isentropic, no loss model)', '0.000', '0.0000', '0.000']
        )
        loss_rows.append(
            [component, 'Subtotal (s_out - s_in)', '0.000', '0.0000', '0.000']
        )
        continue

    for ds_spec, zeta_spec, effdef_spec in zip(
        loss_coeff.ds_specs, loss_coeff.zeta_specs, eff_deficit.effdef_specs
    ):
        label = ds_spec.symbol.removeprefix('ds_').capitalize()
        loss_rows.append(
            [
                component,
                label,
                f'{_val(ds_spec.at_node(outlet_idx)):.3f}',
                f'{_val(zeta_spec.at_node(outlet_idx)):.4f}',
                f'{_val(effdef_spec.at_node(outlet_idx)):.3f}',
            ]
        )

    loss_rows.append(
        [
            component,
            'Subtotal (s_out - s_in)',
            f'{sum(_val(s.at_node(outlet_idx)) for s in loss_coeff.ds_specs):.3f}',
            f'{_val(loss_coeff.zeta_total_spec.at_node(outlet_idx)):.4f}',
            f'{_val(eff_deficit.effdef_total_spec.at_node(outlet_idx)):.3f}',
        ]
    )

print(
    tabulate(
        loss_rows,
        headers=[
            'Component',
            'Loss mechanism',
            'Ds\n[J/kg/K]',
            'zeta = T_is*Ds/(W_is^2/2)\n[-]',
            'Efficiency deficit\n[%]',
        ],
        tablefmt='github',
    )
)

# --- Overall performance vs. Sauret Table 3 (engine conditions, "Present"
# 1D analysis) and Jones (1996) Table 2 (design goal)
specific_work = _val(n5.oth.EulerWork)

print(f'\nSpecific work: {specific_work / 1e3:.1f} kJ/kg')
print('Sauret (Table 3): 366 kJ/kg')
_bc_p0_tot = float(ntw.system.data.boun_cond[n0.tot.Pressure][0])
_bc_p5_static = float(ntw.system.data.boun_cond[n5.stc.Pressure][0])
print(
    f'Rotor-exit pressure ratio (p0_0 / p5_static): '
    f'{_val(n0.tot.Pressure) / _val(n5.stc.Pressure):.3f} '
    f'(Sauret (Table 4): {_bc_p0_tot / _bc_p5_static:.3f})'
)

# ============================================================
# Stage performance
# ============================================================
eta_ts = _val(n5.ndim.EtaTS)
eta_tt = _val(n5.ndim.EtaTT)
print(f'\nTotal-to-static expansion efficiency (eta_ts, station 0 -> 5): {eta_ts:.4f}')
print(f'Total-to-total expansion efficiency (eta_tt, station 0 -> 5): {eta_tt:.4f}')


# ============================================================
# Velocity triangle plots at each station
# ============================================================
if PLOTS:
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
