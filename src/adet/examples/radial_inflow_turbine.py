"""
Supersonic ORC Radial Inflow turbine (ORCHID, TU Delft) design

State:
------
- Stator basic setup is working
- Most losses are missing
- Missing stator-rotor row gap
- Volute not integrated
- The rotor's leading-edge blade metal angle is kept as-designed; a
  fictitious, zero-length ``SlipGap`` component is inserted immediately
  upstream of the rotor to account for flow slip there (Chen & Baines
  1994) without disturbing that geometry or the usual same-station
  continuity link from the vaneless space. See
  ``adet.components.blade_row.SlipGap`` and
  ``chen1994_optimum_incidence.py``.
"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity
from tabulate import tabulate

from adet.assemblers import CasadiSystem
from adet.components.blade_row import BladeRow, Interspace, SlipGap
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.base_equation import LossApplier
from adet.equations.definitions import BoundaryLayerRatios, IsentropicProperties
from adet.equations.nondimensional import GammaPV
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.losses.rit import (
    StatorProfileLoss,
    EndwallLoss,
    ImpellerIncidenceLoss,
    ImpellerLeakageLoss,
    ImpellerPassageLoss,
)
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_velocity_triangles
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)
PLOTS = True


_n0 = NodeVariables(0)
_n1 = NodeVariables(1)


class AddImpellerLosses(LossApplier):
    """Apply the rotor's passage + leakage + incidence losses to the static
    entropy rise."""

    def residual(
        self,
        s0: _n0.stc.Entropy.Hint,
        s1: _n1.stc.Entropy.Hint,
        ds_profile1: _n1.loss.Ds_profile.Hint,
        ds_leakage1: _n1.loss.Ds_leakage.Hint,
        ds_incidence1: _n1.loss.Ds_incidence.Hint,
        ds_endwall1: _n1.loss.Ds_endwall.Hint,
    ):
        return s1 - (s0 + ds_profile1 + ds_leakage1 + ds_incidence1 + ds_endwall1)


class AddStatorLosses(LossApplier):
    """Apply the stator's profile loss to the static entropy rise."""

    def residual(
        self,
        s0: _n0.stc.Entropy.Hint,
        s1: _n1.stc.Entropy.Hint,
        ds_profile1: _n1.loss.Ds_profile.Hint,
        ds_endwall1: _n1.loss.Ds_endwall.Hint,
    ):
        return s1 - (s0 + ds_profile1 + ds_endwall1)


# |> Machine stations, in the order the components are chained below (each
# component owns two fresh global nodes: inlet, outlet)
STATION_LABELS = {
    0: 'Stator inlet',
    1: 'Stator outlet',
    2: 'Interspace inlet',
    3: 'Interspace outlet',
    4: 'Slip gap inlet',
    5: 'Slip gap outlet',
    6: 'Rotor inlet',
    7: 'Rotor outlet',
}

# |> Stator 0 - 1
n0 = NodeVariables(0)
n1 = NodeVariables(1)
# |> Inters 2 - 3
n2 = NodeVariables(2)
n3 = NodeVariables(3)
# |> Slip gap 4 - 5
n4 = NodeVariables(4)
n5 = NodeVariables(5)
# |> Rotor  6 - 7
n6 = NodeVariables(6)
n7 = NodeVariables(7)

inl = Inlet(
    {
        # *** Total conditions
        n0.tot.Pressure: Quantity(18.1, 'bar'),
        n0.tot.Temperature: Quantity(300, 'degC'),
        # ***
        n0.oth.TotMassFlow: 0.132,
        n0.kin.FlowAngleRel: Quantity(65, 'deg'),
        # *** Geometrical
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

stator = BladeRow(
    'nozzle',
    bound_cond={
        n1.geo.HeightRatio: 1.0,
        n1.geo.RadiusRatio: 0.75,
        n1.geo.Rmid: Quantity(26.1, 'mm'),
        n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
        # *** Outlet
        n1.geo.MetalAngle: Quantity(78, 'deg'),
        # n1.kin.FlowAngleRel: Quantity(78, 'deg'),
        # n1.kin.Mach: 2.2,
        # *** Blades
        n0.geo.ThickByPitch: 0.05,
        n1.geo.ThickByPitch: 0.05,
        n1.geo.NumBlades: 12,
        n1.geo.ChordAx: Quantity(0.01, 'mm'),
        # *** Boundary Layer
        n1.oth.MomByBld: 0.075,
        n1.oth.DispByMom: 2.0,
        n1.oth.DispByHgt: 0.09,
        # *** Shock
    },
    shaft=casing,
    extra_equations={
        AddStatorLosses(): (0, 1),
        ZeroDeviation(): 0,  # No incidence
        # *** Loss + Dependencies (disabled: inviscid check)
        StatorProfileLoss(): (0, 1),
        EndwallLoss(): (0, 1),
        IsentropicProperties(): (0, 1),
        BoundaryLayerRatios(): 1,
        GammaPV(): 0,
        GammaPV(): 1,
    },
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

# Fictitious, zero-length lumped station accounting for flow slip (Chen &
# Baines 1994) at the rotor's leading edge -- see
# ``adet.components.blade_row.SlipGap`` and
# ``chen1994_optimum_incidence.py``. Radius/height are unchanged across it
# (RadiusRatio/HeightRatio default to 1.0), it co-rotates with the rotor
# (same shaft), and it carries a *duplicate* of the rotor's own (real,
# unmodified) blade number and LE metal angle so its slip-corrected
# relative flow angle -- not the rotor's own metal angle -- becomes the
# actual incidence the rotor "sees" (via the normal same-station
# from_previous_node link, exactly as it would inherit from any other
# upstream component).
slip_gap = SlipGap(
    'slip_gap',
    shaft=shaft,
    bound_cond={
        n1.geo.NumBlades: 13,
        n1.geo.MetalAngle: Quantity(45, 'deg'),  # rotor's real LE blade angle
    },
    extra_equations={
        IsentropicLink(): (0, 1),
    },
)

rotor = BladeRow(
    'impeller',
    bound_cond={
        # *** Meridional Geometry
        n1.geo.Rhub: Quantity(8.2, 'mm'),
        n1.geo.Height: Quantity(12.3, 'mm'),
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        # *** Blade Geometry
        n0.geo.BldThick: 0.0003,
        n0.geo.MetalAngle: Quantity(45, 'deg'),
        n1.geo.BldThick: 0.0003,
        n1.geo.ChordAx: Quantity(10.2, 'mm'),
        n1.geo.NumBlades: 13,
        # *** Outlet condition
        n1.stc.Pressure: Quantity(0.443, 'bar'),
        # *** Tip clearance
        n0.geo.TipClearance: Quantity(0.2, 'mm'),
        n1.geo.TipClearance: Quantity(0.2, 'mm'),
    },
    shaft=shaft,
    extra_equations={
        AddImpellerLosses(): (0, 1),
        IsentropicProperties(): (0, 1),
        ImpellerPassageLoss(): (0, 1),
        ImpellerLeakageLoss(): (0, 1),
        ImpellerIncidenceLoss(): (0, 1),
        EndwallLoss(): (0, 1),
    },
)

abs_state = DebugAbstractState('HEOS', 'MM')

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
        slip_gap,
        rotor,
    ],
)

ntw.build()

rtfn = ntw.system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False},
)
x0 = ntw.system.get_guess(
    {
        # NOTE: Keep the incidence loss's fractional-power terms away from
        # the |incidence| = 90 deg singularity during early iterations.
        # Slip gap (fictitious, lumped at the rotor LE): same guess as
        # the rotor-inlet triangle it represents.
        n4.kin.FlowAngleRel: Quantity(48, 'deg').to('rad').magnitude,
        n5.kin.FlowAngleRel: Quantity(48, 'deg').to('rad').magnitude,
        n6.kin.FlowAngleRel: Quantity(48, 'deg').to('rad').magnitude,
        # NOTE: Seed the thermodynamic state at every station close to the
        # known-good solution (from a prior solve without the endwall-loss
        # march) -- the new march equations add several more EOS calls
        # (self.eos(P, s) at each radial station) whose validity domain is
        # narrow for this fluid, so starting far from the physical state
        # risks an out-of-range CoolProp query during early iterations
        n0.stc.Pressure: 18.07e5,
        n0.stc.Temperature: 573.1,
        n0.stc.Entropy: 1159.5,
        n1.stc.Pressure: 1.61e5,
        n1.stc.Temperature: 535.6,
        n1.stc.Entropy: 1177.4,
        n2.stc.Pressure: 1.61e5,
        n2.stc.Temperature: 535.6,
        n2.stc.Entropy: 1177.4,
        n3.stc.Pressure: 1.51e5,
        n3.stc.Temperature: 534.8,
        n3.stc.Entropy: 1177.4,
        n4.stc.Pressure: 1.51e5,
        n4.stc.Temperature: 534.8,
        n4.stc.Entropy: 1177.4,
        n5.stc.Pressure: 1.51e5,
        n5.stc.Temperature: 534.8,
        n5.stc.Entropy: 1177.4,
        n6.stc.Pressure: 1.51e5,
        n6.stc.Temperature: 534.8,
        n6.stc.Entropy: 1177.4,
        n7.stc.Pressure: 0.443e5,
        n7.stc.Temperature: 519.7,
        n7.stc.Entropy: 1183.1,
    },
    fallback=0.5,
)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds(
    {
        # WARN: Force the supersonic solution w/ bounds
        n1.kin.Mach: (1.01, 4.0),
        # NOTE: Thermo bounding stabilizes a lot
        n0.stc.Temperature.Glob: (300, 580),
        n0.stc.Pressure.Glob: (1e3, 1e9),
        # NOTE: Keep the endwall-loss march's intermediate (P, s) EOS calls
        # away from the fluid's validity dome during early iterations
        n1.stc.Pressure: (0.3e5, 20e5),
        n0.stc.Entropy.Glob: (200.0, 3000.0),
        # NOTE: With the rotor outlet static pressure fixed, FlowAngleAbs
        # is free and the problem has multiple roots; bound it around the
        # physical branch (else IPOPT lands on a different angle each run)
        n7.kin.FlowAngleAbs: (
            Quantity(-20, 'deg').to('rad').magnitude,
            Quantity(60, 'deg').to('rad').magnitude,
        ),
        # NOTE: Keep the incidence loss's fractional-power terms away from
        # the |incidence| = 90 deg singularity throughout the search. Also
        # applied to the slip gap's own outlet (n5), which is where
        # OptimalIncidenceRadialInflowTurbine actually determines this
        # angle and whose value the rotor inlet (n6) simply inherits.
        n5.kin.FlowAngleRel: (
            Quantity(0, 'deg').to('rad').magnitude,
            Quantity(90, 'deg').to('rad').magnitude,
        ),
        n6.kin.FlowAngleRel: (
            Quantity(0, 'deg').to('rad').magnitude,
            Quantity(90, 'deg').to('rad').magnitude,
        ),
    }
)

sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)

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
# Loss breakdown: specific entropy increase per loss mechanism,
# grouped by component. Each component's "Subtotal" row is the
# actual solved entropy rise (s_out - s_in) across that component,
# which should match the sum of its mechanism rows -- the
# AddStatorLosses/AddImpellerLosses equations enforce that
# equality, so this also doubles as a consistency check.
# ============================================================
LOSS_MECHANISMS = {
    'Stator': [
        ('Profile', lambda n: n.loss.Ds_profile),
        ('Endwall', lambda n: n.loss.Ds_endwall),
    ],
    'Rotor': [
        ('Profile', lambda n: n.loss.Ds_profile),
        ('Leakage', lambda n: n.loss.Ds_leakage),
        ('Incidence', lambda n: n.loss.Ds_incidence),
        ('Endwall', lambda n: n.loss.Ds_endwall),
    ],
}
COMPONENT_NODES = {
    'Stator': (0, 1),
    'Interspace': (2, 3),
    'Slip gap': (4, 5),
    'Rotor': (6, 7),
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

loss_rows.append(['All components', 'TOTAL (s7 - s0)', f'{ds_total_all:.3f}'])

print(
    tabulate(
        loss_rows,
        headers=['Component', 'Loss mechanism', 'Ds\n[J/kg/K]'],
        tablefmt='github',
    )
)

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
