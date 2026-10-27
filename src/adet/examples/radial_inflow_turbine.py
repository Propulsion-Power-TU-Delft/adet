"""
Supersonic ORC Radial Inflow turbine (ORCHID, TU Delft) design

State:
------
- Missing stator-rotor row gap loss (currently purely isentropic, free-vortex flow)
- Volute not integrated
"""

import logging
import math

import CoolProp
import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
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
from adet.equations.geometrical import MeridionalChord
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
from adet.losses.mixing import (
    DentonBaumgartnerMixingLoss,
    DentonMixingLoss,
    SieverdingBasePressure,
)
from adet.losses.rit import (
    StatorProfileLoss,
    StanitzSlip,
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
from adet.equations.utils import residual_debugger

logger = logging.getLogger(__name__)
setup_logger(logger)
PLOTS = True
USE_SLIP_MODEL = True

_n0 = NodeVariables(0)
_n1 = NodeVariables(1)


class AddImpellerLosses(LossApplier):
    """Apply the rotor's passage + leakage + incidence + endwall losses to
    the static entropy rise."""

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
    """Apply the stator's profile + endwall losses to the static entropy
    rise."""

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
    0: 'Stator inlet',
    1: 'Stator outlet',
    2: 'Interspace inlet',
    3: 'Interspace outlet',
    4: 'Rotor inlet',
    5: 'Rotor outlet',
}

# |> Stator 0 - 1
n0 = NodeVariables(0)
n1 = NodeVariables(1)
# |> Inters 2 - 3
n2 = NodeVariables(2)
n3 = NodeVariables(3)
# |> Rotor  4 - 5
n4 = NodeVariables(4)
n5 = NodeVariables(5)

inl = Inlet(
    {
        n0.tot.Pressure: Quantity(18.1, 'bar'),
        n0.tot.Temperature: Quantity(300, 'degC'),
        n0.oth.TotMassFlow: 0.132,
        n0.kin.FlowAngleRel: Quantity(65, 'deg'),
        n0.geo.Height: Quantity(2, 'mm'),
        n0.geo.MeridionalAngle: Quantity(-90, 'deg'),
    }
)

casing = Shaft(
    Quantity(0, 'rpm'),
    is_constrained=True,
)
shaft = Shaft(
    Quantity(98100, 'rpm'),
    is_constrained=True,
)


def build_network(include_losses: bool) -> ComponentNetwork:
    """Build the stator/interspace/rotor network.

    With ``include_losses=False`` every entropy-generating loss mechanism is
    replaced by an ``IsentropicLink`` (s_out = s_in), giving a much
    easier-to-solve system whose solution is used as the initial guess for
    the full, lossy system (see below).
    """
    stator_extra_equations: dict = {
        ZeroDeviation(): 0,
        MeridionalChord(): (0, 1),
    }  # No incidence
    rotor_extra_equations: dict = {MeridionalChord(): (0, 1)}
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
            n1.geo.HeightRatio: 1.0,
            n1.geo.RadiusRatio: 0.75,
            n1.geo.Rmid: Quantity(26.1, 'mm'),
            n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
            n1.geo.MetalAngle: Quantity(78, 'deg'),
            n0.geo.ThickByPitch: 0.05,
            n1.geo.ThickByPitch: 0.05,
            n1.geo.NumBlades: 12,
            # Planar stator: zero axial length, meridional chord reduces
            # to the inlet/outlet radius change
            n1.geo.AxialLength: Quantity(0.0, 'mm'),
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
            # Rotor inlet (Rmid = 25.75 mm) / stator outlet (Rmid = 26.1 mm)
            n1.geo.RadiusRatio: 25.75 / 26.1,
            n1.geo.HeightRatio: 1.0,
        },
        extra_equations={
            IsentropicLink(): (0, 1),
        },
    )

    rotor = BladeRow(
        'impeller',
        bound_cond={
            n1.geo.Rhub: Quantity(8.2, 'mm'),
            n1.geo.Height: Quantity(12.3, 'mm'),
            n1.geo.MeridionalAngle: Quantity(0, 'deg'),
            n0.geo.BldThick: 0.0003,
            n0.geo.MetalAngle: Quantity(45, 'deg'),
            n1.geo.BldThick: 0.0003,
            # Rotor axial length (meridional chord then follows from the
            # vector composition with the midspan radius change)
            n1.geo.AxialLength: Quantity(10.2, 'mm'),
            n1.geo.NumBlades: 13,
            n1.stc.Pressure: Quantity(0.443, 'bar'),
            n0.geo.TipClearance: Quantity(0.2, 'mm'),
            n1.geo.TipClearance: Quantity(0.2, 'mm'),
            n1.oth.MomByBld: 0.075,
            n1.oth.DispByMom: 2.0,
            n1.oth.DispByHgt: 0.09,
        },
        shaft=shaft,
        extra_equations=rotor_extra_equations,
    )

    if USE_SLIP_MODEL:
        # ``ChenSlip`` sets the rotor's own inlet tangential velocity (hence
        # its relative flow angle) directly, so it must stop being a hard
        # copy of the interspace's exit value -- the usual ``BladeRow``
        # behavior via ``ABSOLUTE_LINK``.
        rotor._from_prev_node.discard(KinematicVariables(DEF_NODE).V_tan)

    abs_state = DebugAbstractState('HEOS', 'MM')

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
# Part 1: isentropic initialization (no loss models)
# ============================================================
ntw_isen = build_network(include_losses=False)
ntw_isen.build()

# ============================================================
# Inlet entropy and isentropic rotor-exit static temperature, from the
# inlet boundary conditions and the rotor exit static pressure
# ============================================================
_isen_bc = ntw_isen.system.data.boun_cond
p0_tot = float(_isen_bc[n0.tot.Pressure][0])
t0_tot = float(_isen_bc[n0.tot.Temperature][0])
p5_static = float(_isen_bc[n5.stc.Pressure][0])

_bounds_state = cp.AbstractState('HEOS', 'MM')
_bounds_state.update(cp.PT_INPUTS, p0_tot, t0_tot)
ENTROPY_IN = _bounds_state.smass()

_bounds_state.update(cp.PSmass_INPUTS, p5_static, ENTROPY_IN)
T5_ISENTROPIC = _bounds_state.T()

# ============================================================
# Bounds
# ============================================================
BOUNDS = {
    n0.stc.Temperature.Glob: (T5_ISENTROPIC, 600),
    n0.stc.Pressure.Glob: (0.1e5, 30e5),
    n1.kin.Mach: (1.5, 2.5),
    # n4.kin.FlowAngleRel: (
    #     Quantity(0, 'deg').to('rad').magnitude,
    #     Quantity(60, 'deg').to('rad').magnitude,
    # ),
    # n5.kin.FlowAngleAbs: (
    #     Quantity(-20, 'deg').to('rad').magnitude,
    #     Quantity(40, 'deg').to('rad').magnitude,
    # ),
}

rtfn_isen = ntw_isen.system.make_rootfinder('ipopt', opts={'error_on_fail': False})
x0_isen = ntw_isen.system.get_guess(fallback=0.5)
kn_isen = ntw_isen.system.get_boundary_conds()
bnd_isen = ntw_isen.system.get_bounds(BOUNDS)

sol_isen = solve_root_problem(
    rtfn_isen, x0_isen, kn_isen, bnd_isen, suppress_output=True
)
rtfn_isen = ntw_isen.system.make_rootfinder('kinsol')
sol_isen = solve_root_problem(rtfn_isen, sol_isen, kn_isen, bnd_isen)

isen_data = ntw_isen.system.sol_to_dict(sol_isen)
logger.info('Isentropic initialization solved; seeding the real system with it.')

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


_work_is_guess = isen_data[n0.tot.Enthalpy] - isen_data[n5.tot.Enthalpy]
isen_property_guess[n5.oth.EulerWorkIs] = _work_is_guess
isen_property_guess[n5.oth.EulerWork] = _work_is_guess

# ============================================================
# Part 2: real solution (with loss models)
# ============================================================

ntw = build_network(include_losses=True)
ntw.system.add_equation(TotalStaticExpansionEfficiency(), (0, 5))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 5))
ntw.system.add_equation(EulerWork(), (0, 5))
ntw.system.add_equation(IsentropicEulerWork(), (0, 5))

# ============================================================
# Add one LossCoefficient/EfficiencyDeficit equation per component
# ============================================================
COMPONENT_NODES = {
    'Stator': (0, 1),
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

BOUNDS = {
    **BOUNDS,
    # n5.stc.Entropy: (ENTROPY_IN - 200, ENTROPY_IN),
    n5.stc.Temperature: (T5_ISENTROPIC, t0_tot),
}

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
        # f'{_val(n.geo.RDistr) * 1e3:.2f}',
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
            # 'R [mm]',
        ],
        tablefmt='github',
    )
)

# ============================================================
# Loss breakdown: Denton-style loss coefficient (zeta) and the resulting
# efficiency deficit, per mechanism
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
