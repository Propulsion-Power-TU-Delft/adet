"""Define some components"""

from pint import Quantity

# Equations
from adet.equations.nondimensional import WorkCoefficient
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation

# Tooling & Components
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
        'Vt_mid': 'm/s',  # For vortex distributions
        'alpha_mid': 'rad',  # For vortex distributions
        'sizeParameter': 'meters',
        'n_blades': 'dimensionless',
        # Profile losses
        'Cd_profile': 'dimensionless',
        'xi_by_camb_len_A': 'meters',
        'xi_by_camb_len_B': 'meters',
        'k_prof': 'dimensionless',
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
            'alpha': Quantity(0, 'deg'),
            'Vm': Quantity(80, 'm/s'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'height': 0.2,
        },
        'tot': {
            'p': 6e5,  # impose at outlet
            'T': 700,
        },
        'oth': {
            # 'cum_massflow': 90,
        },
    }
)

row0 = BladeRow(
    'Stator',
    {
        'kin': {
            # 'alpha': Quantity(70, 'deg'),
        },
        'geo': {
            # Meridional
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            # Blade
            'chord_ax': 0.15,
            'n_blades': 25,
            # 'solidity': 0.4,
        },
        'tot': {
            # 'p': 6e5, # Impose either here or at inlet
        },
        'oth': {
            'heightRatio': 1.1,
            'mach': 0.3,
        },
    },
    shaft=static_shaft,
    extra_equations={
        # |> Losses & Dev
        PercentageEntropyLoss(0.0): (0, 1),
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        # MidspanAngle(): 1,
        # TotalPressureLoss(0.0): (0, 1),
        # DentonProfileLoss(real_model): (0, 1),
    },
)

row1 = BladeRow(
    'Rotor',
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
            'n_blades': 22,
        },
        'tot': {
            # 'p': 6e5, # Impose either here or at inlet
        },
        'oth': {
            'heightRatio': 1.1,
            'workCoeff': 1.5,
        },
    },
    shaft=rotating_shaft,
    extra_equations={
        # |> Losses & Dev
        PercentageEntropyLoss(0.0): (0, 1),
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        WorkCoefficient(): (0, 1),
        # MidspanAngle(): 1,
        # TotalPressureLoss(0.0): (0, 1),
        # DentonProfileLoss(real_model): (0, 1),
    },
)
