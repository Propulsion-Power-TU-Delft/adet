"""
Sundstrand T-100 MPSPU radial-inflow turbine (Jones, 1996 / Sauret, 2012)

Reproduces the same nozzle -> vaneless interspace -> rotor architecture as
``radial_inflow_turbine.py``, applied to the 50 hp engine design point of
the Sundstrand Power Systems T-100 Multipurpose Small Power Unit turbine,
described in:

    A. C. Jones, "Design and Test of a Small, High Pressure Ratio Radial
    Turbine," ASME J. Turbomach., vol. 118, pp. 362-370, April 1996.
    (``docs/Jones1996 JoT.pdf``)

    E. Sauret, "Open Design of High Pressure Ratio Radial-Inflow Turbine
    for Academic Validation," Proc. ASME IMECE2012, IMECE2012-88315.
    (``docs/Sauret2012 Open Design T100.pdf``)

Jones (1996) gives the cycle boundary conditions (Table 2) and mean-line
velocity triangles (Fig. 4); its geometry is only available as scanned
drawings. Sauret (2012) independently tabulates the exact 3D geometry of
this same turbine (her Tables 1-2 and Appendix A), so geometry here is
taken from Sauret wherever available, falling back to Jones' text only
for the few items Sauret doesn't tabulate (nozzle inlet swirl angle).
Every geometric/boundary-condition parameter is commented with its
source table/figure; items neither paper tabulates are flagged
``NOTE (assumed)``.

Jones' Table 2 "Specific work" (43.9 BTU/lb) is not used as a boundary
condition -- it disagrees by roughly a factor of 4 with both Jones' own
tip-speed/Vtheta data and Sauret's Table 3 Power figure; printed for
reference only. Rotor exit static pressure is taken from Sauret's Table 4
"Present" (RITAL) result at engine conditions, not Jones' rig-test Table 3.

State:
------
- Design point only (50 hp rating); the 75 hp uprating is not modeled.
- No exhaust diffuser (rotor exit is the last modeled station).
- Zero-deviation at every row exit, as in ``radial_inflow_turbine.py``.
- A fictitious, zero-length ``SlipGap`` component upstream of the rotor
  accounts for rotor-inlet flow slip (Chen & Baines 1994) -- see
  ``adet.components.blade_row.SlipGap``.
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
from adet.losses.mixing import DentonBaumgartnerMixingLoss, SieverdingBasePressure
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
    4: 'Slip gap inlet',
    5: 'Slip gap outlet',
    6: 'Rotor inlet',
    7: 'Rotor outlet',
}

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)
n4 = NodeVariables(4)
n5 = NodeVariables(5)
n6 = NodeVariables(6)
n7 = NodeVariables(7)

# ============================================================
# Geometry and boundary conditions. Radii/heights/lengths/angles are
# Sauret Table 2 and Appendix A (Tables 6-11) unless noted otherwise;
# cycle boundary conditions are Sauret Table 1.
# ============================================================

# Nozzle radii [m] (Sauret Table 2, "Nozzle")
NOZZLE_R_IN = 0.074
NOZZLE_R_OUT = 0.0635
NOZZLE_RADIUS_RATIO = NOZZLE_R_OUT / NOZZLE_R_IN  # n1(exit)/n0(inlet)

# Nozzle heights [m] (Sauret Table 2)
NOZZLE_HEIGHT_IN = 0.00635
NOZZLE_HEIGHT_OUT = 0.006
NOZZLE_HEIGHT_RATIO = NOZZLE_HEIGHT_OUT / NOZZLE_HEIGHT_IN

# Nozzle blade geometry (Sauret Table 2 "Nozzle" + Appendix Table 6, m=0%)
NOZZLE_TE_THICKNESS = 0.00051
NOZZLE_LE_THICKNESS = 0.000661
NOZZLE_CHORD = 0.0229
NOZZLE_NUM_BLADES = 19  # Sauret Table 1 / Jones text

# Rotor inlet (inducer) tip radius [m] (Sauret Table 2, "Rotor" R_in);
# matches Jones' "rotor tip diameter of 4.58 inches" to within rounding.
R2_TIP = 0.0582

# Vaneless-space (nozzle exit -> rotor inlet) radius/height ratios, from
# the two rows' tabulated radii/heights above (Sauret Table 2)
VANELESS_GAP_RATIO = R2_TIP / NOZZLE_R_OUT  # rotor inlet R / nozzle exit R
VANELESS_HEIGHT_RATIO = 0.00635 / NOZZLE_HEIGHT_OUT

# Exducer (rotor exit) hub/tip radii [m] (Sauret Table 8, m=100%)
R3_HUB = 0.01524
R3_TIP = 0.03683
H3 = R3_TIP - R3_HUB

# Rotor exit mean relative flow angle (Sauret Fig. 8(c), Present/RITAL)
BETA3_MEAN = Quantity(-57.3, 'deg')

# Rotor blade geometry (Sauret Table 2 "Rotor" + Appendix Tables 9-10)
ROTOR_AXIAL_LENGTH = 0.0389
ROTOR_TE_THICKNESS = 0.00076
ROTOR_LE_THICKNESS = 0.00086
ROTOR_CHORD = 0.0457
ROTOR_INLET_CLEARANCE = 0.0004
ROTOR_EXDUCER_CLEARANCE = 0.00023
ROTOR_NUM_BLADES = 16  # Sauret Table 1 / Jones text

# Cycle boundary conditions at the engine design point (Sauret Table 1;
# matches Jones' Table 2 50 hp rating)
T0_TOT = Quantity(1056.5, 'K')
P0_TOT = Quantity(580400, 'Pa')
MDOT = Quantity(0.33, 'kg/s')
N_RPM = Quantity(106588, 'rpm')

# Rotor exit static pressure: Sauret Table 4, "Present" (RITAL) column,
# engine conditions (not Jones' rig-test Table 3, see module docstring)
P5_STATIC = 94700.0

# NOTE (assumed / Jones-only): Sauret doesn't tabulate nozzle inlet
# swirl; Jones gives a 20-40 deg range, midpoint used here.
NOZZLE_INLET_SWIRL = Quantity(30.0, 'deg')

# Nozzle exit angle (Sauret Fig. 8(a), Present/RITAL)
NOZZLE_EXIT_ANGLE = Quantity(78.1, 'deg')

inl = Inlet(
    {
        n0.tot.Pressure: P0_TOT,
        n0.tot.Temperature: T0_TOT,
        n0.oth.TotMassFlow: MDOT.magnitude,
        n0.kin.FlowAngleRel: NOZZLE_INLET_SWIRL,
        n0.geo.Height: Quantity(NOZZLE_HEIGHT_IN, 'm'),
        n0.geo.MeridionalAngle: Quantity(-90, 'deg'),
    }
)

casing = Shaft(Quantity(0, 'rpm'), is_constrained=True)
shaft = Shaft(N_RPM, is_constrained=True)

stator = BladeRow(
    'nozzle',
    bound_cond={
        n1.geo.HeightRatio: NOZZLE_HEIGHT_RATIO,
        n1.geo.RadiusRatio: NOZZLE_RADIUS_RATIO,
        n1.geo.Rmid: Quantity(NOZZLE_R_OUT, 'm'),
        n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
        n1.geo.MetalAngle: NOZZLE_EXIT_ANGLE,
        n0.geo.BldThick: Quantity(NOZZLE_LE_THICKNESS, 'm'),
        n1.geo.BldThick: Quantity(NOZZLE_TE_THICKNESS, 'm'),
        n1.geo.NumBlades: NOZZLE_NUM_BLADES,
        # Negligible axial chord: purely radial nozzle passage (~0 true
        # axial extent), same simplification as radial_inflow_turbine.py.
        n1.geo.MerChord: Quantity(0.01, 'mm'),
        # Assumed: not given in either paper.
        n1.oth.MomByBld: 0.075,
        n1.oth.DispByMom: 2.0,
        n1.oth.DispByHgt: 0.09,
    },
    shaft=casing,
    extra_equations={
        AddStatorLosses(): (0, 1),
        ZeroDeviation(): 0,  # No incidence
        StatorProfileLoss(): (0, 1),
        EndwallLoss(): (0, 1),
        IsentropicProperties(): (0, 1),
        BoundaryLayerRatios(): 1,
        GammaPV(): 0,
        GammaPV(): 1,
        SieverdingBasePressure(): (0, 1),
        DentonBaumgartnerMixingLoss(): 1,
    },
)

interspace = Interspace(
    'intrspc',
    {
        n1.geo.RadiusRatio: VANELESS_GAP_RATIO,
        n1.geo.HeightRatio: VANELESS_HEIGHT_RATIO,
    },
    extra_equations={
        IsentropicLink(): (0, 1),
    },
)

# Fictitious, zero-length lumped station for rotor-inlet flow slip (Chen &
# Baines 1994) -- see adet.components.blade_row.SlipGap.
slip_gap = SlipGap(
    'slip_gap',
    shaft=shaft,
    bound_cond={
        n1.geo.NumBlades: ROTOR_NUM_BLADES,
        n1.geo.MetalAngle: Quantity(0, 'deg'),  # rotor's real LE blade angle
    },
    extra_equations={
        IsentropicLink(): (0, 1),
    },
    correlation='chen',
)

rotor = BladeRow(
    'impeller',
    bound_cond={
        # *** Meridional geometry (exducer, node 1 = rotor outlet)
        n1.geo.Rhub: Quantity(R3_HUB, 'm'),
        n1.geo.Height: Quantity(H3, 'm'),
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),  # axial exducer exit
        # *** Blade geometry
        n0.geo.BldThick: Quantity(ROTOR_LE_THICKNESS, 'm'),
        n0.geo.MetalAngle: Quantity(0, 'deg'),  # purely radial LE blades
        n1.geo.BldThick: Quantity(ROTOR_TE_THICKNESS, 'm'),
        n1.geo.MerChord: Quantity(ROTOR_AXIAL_LENGTH, 'm'),
        n1.geo.NumBlades: ROTOR_NUM_BLADES,
        n1.stc.Pressure: Quantity(P5_STATIC, 'Pa'),
        n0.geo.TipClearance: Quantity(ROTOR_INLET_CLEARANCE, 'm'),
        n1.geo.TipClearance: Quantity(ROTOR_EXDUCER_CLEARANCE, 'm'),
        # Assumed: not given in either paper, same as the nozzle exit.
        n1.oth.MomByBld: 0.075,
        n1.oth.DispByMom: 2.0,
        n1.oth.DispByHgt: 0.09,
    },
    shaft=shaft,
    extra_equations={
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
    },
)

# Assumed: neither paper gives gas composition/property data for the
# T-100's combustion-product working fluid; approximated with air.
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
        slip_gap,
        rotor,
    ],
)

ntw.build()

rtfn = ntw.system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False},
)

# ============================================================
# Initial guess -- Sauret Table 4's "Present" (RITAL) 1D meanline P_S/T_S
# per station at engine conditions, with matching CoolProp entropy values
# (a tight guess is needed since the real-fluid EOS has a narrow validity
# domain).
# ============================================================
U2 = 649.224  # rotor inlet tip speed [m/s] (= 2130 ft/s, Jones Fig. 2(a))
VTHETA4_OVER_U = 0.882  # Jones Fig. 4 inlet triangle
# Rotor inlet relative flow angle (Sauret Fig. 8(b), Present/RITAL);
# blades are purely radial at the LE, so this is also the incidence.
BETA4_GUESS = Quantity(-32.7, 'deg')

x0 = ntw.system.get_guess(
    {
        n4.kin.FlowAngleRel: BETA4_GUESS.to('rad').magnitude,
        n5.kin.FlowAngleRel: BETA4_GUESS.to('rad').magnitude,
        n6.kin.FlowAngleRel: BETA4_GUESS.to('rad').magnitude,
        n7.kin.FlowAngleRel: BETA3_MEAN.to('rad').magnitude,
        n0.stc.Pressure: 5.78e5,
        n0.stc.Temperature: 1050.0,
        n0.stc.Entropy: 4708.1,
        n1.stc.Pressure: 3.284e5,
        n1.stc.Temperature: 931.8,
        n1.stc.Entropy: 4735.6,
        n2.stc.Pressure: 3.284e5,
        n2.stc.Temperature: 931.8,
        n2.stc.Entropy: 4735.6,
        n3.stc.Pressure: 2.948e5,
        n3.stc.Temperature: 906.7,
        n3.stc.Entropy: 4735.8,
        n4.stc.Pressure: 2.948e5,
        n4.stc.Temperature: 906.7,
        n4.stc.Entropy: 4735.8,
        n5.stc.Pressure: 2.948e5,
        n5.stc.Temperature: 906.7,
        n5.stc.Entropy: 4735.8,
        n6.stc.Pressure: 2.948e5,
        n6.stc.Temperature: 906.7,
        n6.stc.Entropy: 4735.8,
        n7.stc.Pressure: P5_STATIC,
        n7.stc.Temperature: 713.9,
        n7.stc.Entropy: 4798.9,
    },
    fallback=0.5,
)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds(
    {
        # Nozzle throat is close to choke (Sauret Table 4: M = 0.887);
        # keep the solver away from a supersonic branch.
        n1.kin.Mach: (0.3, 1.05),
        n0.stc.Temperature.Glob: (300, 1300),
        n0.stc.Pressure.Glob: (1e3, 2e6),
        n1.stc.Pressure: (0.3e5, 8e5),
        n0.stc.Entropy.Glob: (3000.0, 7000.0),
        # Near-zero design exit swirl (Jones Fig. 4: V_theta/U = 0.014)
        n7.kin.FlowAngleAbs: (
            Quantity(-20, 'deg').to('rad').magnitude,
            Quantity(20, 'deg').to('rad').magnitude,
        ),
        # Rotor inlet incidence is negative (see BETA4_GUESS); bound
        # around that branch, away from the ImpellerIncidenceLoss
        # |incidence| = 90 deg singularity.
        n5.kin.FlowAngleRel: (
            Quantity(-70, 'deg').to('rad').magnitude,
            Quantity(0, 'deg').to('rad').magnitude,
        ),
        n6.kin.FlowAngleRel: (
            Quantity(-70, 'deg').to('rad').magnitude,
            Quantity(0, 'deg').to('rad').magnitude,
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

# --- Overall performance vs. Sauret Table 3 (engine conditions, "Present"
# 1D analysis) and Jones (1996) Table 2 (design goal)
h0 = _val(n0.tot.Enthalpy)
h5 = _val(n7.tot.Enthalpy)
specific_work = h0 - h5

print(
    f'\nSpecific work (h0_tot - h5_tot): {specific_work:.1f} J/kg '
    f'({specific_work / 2326.0:.1f} BTU/lb)'
)
print('Sauret Table 3 gives Power = 120.8 kW at 0.33 kg/s (=~ 366 kJ/kg =~ 157 BTU/lb)')
print(
    f'Rotor-exit pressure ratio (p0_0 / p5_static): '
    f'{_val(n0.tot.Pressure) / _val(n7.stc.Pressure):.3f} '
    f'(Sauret Table 4, Present: {P0_TOT.magnitude / P5_STATIC:.3f})'
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
