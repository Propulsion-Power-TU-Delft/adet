"""Define some components"""

from pint import Quantity

from adet.registries import DefaultUnitsRegistry, ScalingRegistry, GuessRegistry
from adet.fluid.settings import AbstractStateModel, IdealGasModel
from adet.components import BladeRow, Shaft, Inlet
from adet.equations.fundamental import BladeCount, ParabolicCamberline
from adet.equations.simplelosses import ZeroDeviation
from adet.losses.profile import DentonProfileLoss, RectVelocityIncompressible
from adet.losses.basic import PercentageEntropyLoss
from adet.equations.nondimensional import (
    StaticTotalPressRatio,
    WorkCoefficient,
    FlowCoefficient,
    SizeParameter,
    SpecificSpeed,
)
from adet.equations.definitions import AngleDeflection
from adet.tools.coolprop_utils import DebugAbstractState

# fluid_model = IdealGasModel(287.0, 1.4)

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('HEOS', 'Air')
abs_state.debug_print = True
fluid_model = AbstractStateModel(abs_state)

# *** Shafts
static_shaft = Shaft(0.0)
rotating_shaft = Shaft(Quantity(1000, 'rpm'))

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

# COMPONENT STACK
inlet = Inlet(
    {
        'kin': {
            # 'alpha': Quantity(25, 'deg'),
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
            'flowCoeff': 1.3,
            # 'cum_massflow': 100,
        },
    }
)

row1 = BladeRow(
    {
        'kin': {
            # 'alpha': Quantity(0, 'deg'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.45,
            'height': 0.2,
            # 'camb_len': 0.2,
            # 'stagger': 0.6,
            'n_blades': 10,
            'chord': 0.2,
            # 'pitch': 0.2,
        },
        'stc': {
            # 'p': 2e5,
        },
        'oth': {
            # PROFILE LOSSES
            'Cd_profile': 0.002,
            # Denton
            'x_by_camb_len_A': 0.375,
            'x_by_camb_len_B': 0.675,
            # NONDIMENSIONAL
            # 'STratio': 0.98,
            'workCoeff': 1.2,
            # These two are not tested
            # You can check plausible values
            # 'specificSpeed': 0.4,
            # 'sizeParameter': 0.1,
        },
    },
    rotating_shaft,
    loss_models=[
        # PercentageEntropyLoss(0.0),
    ],
    extra_equations={
        ZeroDeviation(): 1,  # No outlet deviation
        RectVelocityIncompressible(): (0, 1),  # Rectangular profile
        # DentonProfileLoss(fluid_model): (0, 1),  # Rectangular profile
        ParabolicCamberline(): (0, 1),
        BladeCount(): 1,
        # -| Compute nondimensional coefficients |-
        FlowCoefficient(): 0,
        WorkCoefficient(): (0, 1),
        SpecificSpeed(): (0, 1),
        SizeParameter(): (0, 1),
        StaticTotalPressRatio(): (0, 1),
    },
)

row2 = BladeRow(
    {
        'kin': {
            'meridional_angle': Quantity(0.0, 'deg'),
            'rmid': 1.0,
            'height': 0.3,
        },
        'oth': {
            'workCoeff': 1.1,
            # 'deflection': Quantity(65, 'deg'),
        },
    },
    rotating_shaft,
    loss_models=[
        PercentageEntropyLoss(0.0),
    ],
    extra_equations={
        WorkCoefficient(): (0, 1),
        # AngleDeflection(): (0, 1),
    },
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
    extra_equations={
        # WorkCoefficient(): (0, 1),
        AngleDeflection(): (0, 1),
    },
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
    extra_equations={
        WorkCoefficient(): (0, 1),
    },
)
