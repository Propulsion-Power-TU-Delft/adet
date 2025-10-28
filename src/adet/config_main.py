"""Define some components"""

from pint import Quantity

from adet.registries import DefaultUnitsRegistry, ScalingRegistry, GuessRegistry
from adet.fluid.settings import ExternalFluidModel, IdealGasModel
from adet.components import BladeRow, Shaft, Inlet
from adet.equations.fundamental import BladeCount, ParabolicCamberline
from adet.equations.simplelosses import ZeroDeviation
from adet.losses.profile import DentonProfileLoss, RectVelocityIncompressible
from adet.equations.simplelosses import PercentageEntropyLoss
from adet.equations.nondimensional import (
    StaticTotalPressRatio,
    WorkCoefficient,
    FlowCoefficient,
    SizeParameter,
    SpecificSpeed,
)
from adet.equations.definitions import AngleDeflection
from adet.tools.coolprop_utils import DebugAbstractState


# This counts the number of updates in an attribute
abs_state = DebugAbstractState('HEOS', 'Air')
abs_state.debug_print = False

real_model = ExternalFluidModel(abs_state)
ideal_model = IdealGasModel()


# Set custom units and defaults
_dfu_reg = DefaultUnitsRegistry()
_scl_reg = ScalingRegistry()
_gss_reg = GuessRegistry()

# Add units for some variables
_dfu_reg.from_dict(
    {
        'delta_smass_pct': 'J/(kg*K)',
        'deflection': 'rad',
        'percentage_loss': 'dimensionless',
        'workCoeff': 'dimensionless',
        'flowCoeff': 'dimensionless',
        'specificSpeed': 'dimensionless',
        'STratio': 'dimensionless',
        'Cd_profile': 'dimensionless',
        'sizeParameter': 'meters',
        'n_blades': 'dimensionless',
        'x_by_camb_len_A': 'meters',
        'x_by_camb_len_B': 'meters',
        'k_prof': '',
    }
)

# Set default values for scales and guesses to 1.0
_scl_reg.set_fallback_value(1.0)
_gss_reg.set_fallback_value(1.0)

# BLADE ROWS EQUATION STACK
EXTRA_EQUATIONS = {
    ZeroDeviation(): 1,  # No outlet deviation
    ParabolicCamberline(): (0, 1),
    BladeCount(): 1,
    # -| Compute nondimensional coefficients |-
    WorkCoefficient(): (0, 1),
    FlowCoefficient(): 0,
    ## Profile Losses
    # RectVelocityIncompressible(): (0, 1),  # Rectangular profile
    # DentonProfileLoss(real_model): (0, 1),
}

# *** Shafts
static_shaft = Shaft(0.0)
rotating_shaft = Shaft(Quantity(1000, 'rpm'))

# COMPONENT STACK
inlet = Inlet(
    {
        'kin': {
            'Vm': 100,
            'beta': Quantity(0, 'deg'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'height': 0.15,
        },
        'tot': {
            'p': 3e5,
            'T': 500,
        },
        'oth': {
            # 'cum_massflow': 90,
        },
    }
)
row1 = BladeRow(
    {
        'kin': {
            'alpha': Quantity(10, 'deg'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'height': 0.18,
            # Blades
            'n_blades': 13,
            # 'pitch': 0.15,
            'chord': 0.2,
        },
        'stc': {
            # 'p': 2e5,
        },
        'oth': {
            # PROFILE LOSSES
            # BL coefficient
            # k_prof can act as loading criteria
            # But results in varying blade number
            # along the span if imposed alone
            # 'k_prof': 0.6,
            # Denton
            # 'workCoeff': 1.8,
            # NONDIMENSIONAL
            # 'STratio': 0.98,
            # These two are not tested
            # You can check plausible values
            # 'specificSpeed': 0.4,
            # 'sizeParameter': 0.1,
        },
    },
    static_shaft,
    loss_models=[],
    extra_equations={**EXTRA_EQUATIONS, PercentageEntropyLoss(0.0): (0, 1)},
)

row2 = BladeRow(
    {
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'height': 0.2,
            'n_blades': 13,
            # 'pitch': 0.15,
            'chord': 0.2,
        },
        'stc': {
            # 'p': 2e5,
        },
        'oth': {
            # PROFILE LOSSES
            # BL coefficient
            # k_prof can act as loading criteria
            # But results in varying blade number
            # along the span if imposed alone => Implement Zweifel
            # 'k_prof': 0.6,
            # Denton
            'workCoeff': 1.8,
            'flowCoeff': 1.3,
            # NONDIMENSIONAL
            # 'STratio': 0.98,
            # These two are not tested
            # You can check plausible values
            # 'specificSpeed': 0.4,
            # 'sizeParameter': 0.1,
        },
    },
    rotating_shaft,
    loss_models=[],
    extra_equations={**EXTRA_EQUATIONS, DentonProfileLoss(real_model): (0, 1)},
)


row3 = BladeRow(
    {
        'kin': {
            'meridional_angle': Quantity(-70.0, 'deg'),
            # 'alpha': Quantity(65.0, 'deg'),
            'rmid': 0.8,
            'height': 0.35,
        },
        'oth': {
            # 'workCoeff': 1.0,
            'deflection': Quantity(100.0, 'deg'),
        },
    },
    static_shaft,
    loss_models=[
        PercentageEntropyLoss(0.0),
    ],
    extra_equations=EXTRA_EQUATIONS,
)

row4 = BladeRow(
    {
        'kin': {
            'meridional_angle': Quantity(0.0, 'deg'),
            'rmid': 0.4,
            'height': 0.5,
        },
        'oth': {
            'workCoeff': 1.0,
        },
    },
    rotating_shaft,
    loss_models=[
        PercentageEntropyLoss(0.0),
    ],
    extra_equations=EXTRA_EQUATIONS,
)
