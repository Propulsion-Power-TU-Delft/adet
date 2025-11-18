"""Define some components"""

from pint import Quantity

# Equations
from adet.components.blade_row import DownstreamMixer
from adet.equations.fundamental import BladeBlockage, ZeroBlockage
from adet.equations.geometrical import (
    MinimalCamberLine,
    ParabolicCamberline,
    TwoSegmentCamberline,
)
from adet.equations.nondimensional import WorkCoefficient
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation

# Tooling & Components
from adet.equations.definitions import BladePitchCount, BoundaryLayerRatios
from adet.losses.profile import DentonProfileLoss
from adet.tools.coolprop_utils import DebugAbstractState
from adet.registries import DefaultUnitsRegistry, ScalingRegistry, GuessRegistry
from adet.fluid.settings import ExternalFluidModel, IdealGasModel
from adet.components import BladeRow, Shaft, Inlet


# This counts the number of updates in an attribute
abs_state = DebugAbstractState('HEOS', 'Air')
abs_state.debug_print = False

real_model = ExternalFluidModel(abs_state)
ideal_model = IdealGasModel()


# Set custom units and defaults
_dfu_reg = DefaultUnitsRegistry()
_scl_reg = ScalingRegistry()
_gss_reg = GuessRegistry()

# Add units for custom variables
_dfu_reg.from_dict(
    {
        'delta_smass_pct': 'J/ (kg * K)',
        'deflection': 'rad',
        'percentage_loss': 'dimensionless',
        'workCoeff': 'dimensionless',
        'flowCoeff': 'dimensionless',
        'specificSpeed': 'dimensionless',
        'STratio': 'dimensionless',
        'VmRatio': 'dimensionless',
        'sizeParameter': 'meters',
        'n_blades': 'dimensionless',
        # Profile losses
        'Cd_profile': 'dimensionless',
        'xi_by_camb_len_A': 'meters',
        'xi_by_camb_len_B': 'meters',
        'k_prof': 'dimensionless',
        'mom_by_bld_thick': 'dimensionless',
        'disp_by_mom_thick': 'dimensionless',
    }
)

# Set fallback values for scales and guesses to 1.0
_scl_reg.set_fallback_value(1.0)
_gss_reg.set_fallback_value(1.0)


# *** Shafts
static_shaft = Shaft(
    Quantity(0.0, 'rpm'),
    is_constrained=True,
)
rotating_shaft = Shaft(
    Quantity(1000.0, 'rpm'),
    is_constrained=True,
)

# COMPONENT STACK
inlet = Inlet(
    {
        'kin': {
            'alpha': Quantity(50, 'deg'),
            'mach': 0.1,
            'omega': 0.0,
            # 'Vm': Quantity(80, 'm/s'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'height': 0.2,
        },
        'tot': {
            'p': 6e5,
            'T': 700,
        },
    }
)

row0_mixer = DownstreamMixer(
    'row0_mixer',
    in_constraints={
        'geo': {
            'bld_thick': 0.0025,
            'pitch': 0.126,
        },
        'oth': {
            'disp_thick': 0.0004,
            'mom_thick': 0.0002,
        },
    },
    out_constraints={},
    extra_equations={
        # Add blockage from blades in 0
    },
)

row0 = BladeRow(
    name='Stator',
    shaft=static_shaft,
    in_constraints={
        'geo': {
            'thick_by_pitch': 0.04,
        },
    },
    out_constraints={
        'kin': {
            'alpha': Quantity(60, 'deg'),
            # 'mach': 0.2,
        },
        'geo': {
            # Meridional
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            # Blade
            'chord_ax': 0.15,
            'n_blades': 25,
            'thick_by_pitch': 0.02,  # Blade thickness by pitch
        },
        'tot': {
            # 'p': 5e5,  # Impose either here or at inlet
        },
        'oth': {
            'heightRatio': 1.1,
            'mom_by_bld_thick': 0.075,
            'disp_by_mom_thick': 2,
        },
    },
    extra_equations={
        # Camberline model
        MinimalCamberLine(): (0, 1),
        # TwoSegmentCamberline(): (0, 1),
        # ParabolicCamberline(): (0, 1),
        # |> Losses & Dev
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        PercentageEntropyLoss(0.0): (0, 1),
        # DentonProfileLoss(real_model): (0, 1),
        # |> Boundary layer properties for mixing
        # BoundaryLayerRatios(): 1,
    },
)


row1 = BladeRow(
    'Rotor',
    rotating_shaft,
    {},
    {
        'kin': {
            # 'beta': Quantity(0, 'deg'),
            # 'alpha': 0.0,
        },
        'geo': {
            # Meridional
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            # Blade
            'chord_ax': 0.15,
            'n_blades': 25,
        },
        'tot': {
            # 'p': 6e5, # Impose either here or at inlet
        },
        'oth': {
            'heightRatio': 1.1,
            'workCoeff': 2.4,
        },
    },
    extra_equations={
        # Camberline model
        MinimalCamberLine(): (0, 1),
        # TwoSegmentCamberline(): (0, 1),
        # ParabolicCamberline(): (0, 1),
        # |> Losses & Dev
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        WorkCoefficient(): (0, 1),
        # DentonProfileLoss(real_model): (0, 1),
        PercentageEntropyLoss(0.0): (0, 1),
    },
)
