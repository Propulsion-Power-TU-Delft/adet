"""Define some components"""

from pint import Quantity

from adet.equations.definitions import HeightRatio, MeridionalVelocityRatio
from adet.equations.nondimensional import (
    AbsoluteMachNumber,
    FlowCoefficient,
    WorkCoefficient,
)
from adet.registries import DefaultUnitsRegistry, ScalingRegistry, GuessRegistry
from adet.fluid.settings import ExternalFluidModel, IdealGasModel
from adet.components import BladeRow, Shaft, Inlet
from adet.losses.profile import DentonProfileLoss, RectVelocityIncompressible
from adet.equations.simplelosses import PercentageEntropyLoss
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
        'VmRatio': 'dimensionless',
        'Cd_profile': 'dimensionless',
        'sizeParameter': 'meters',
        'n_blades': 'dimensionless',
        'x_by_camb_len_A': 'meters',
        'x_by_camb_len_B': 'meters',
        'k_prof': '',
    }
)

# Set fallback values for scales and guesses to 1.0
_scl_reg.set_fallback_value(1.0)
_gss_reg.set_fallback_value(1.0)


# *** Shafts
static_shaft = Shaft(0.0)
rotating_shaft = Shaft(Quantity(1000, 'rpm'))

# COMPONENT STACK
inlet = Inlet(
    {
        'kin': {
            # 'Vm': 100,
            'alpha': Quantity(0, 'deg'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'height': 0.15,
        },
        'tot': {
            'p': 6e5,
            'T': 700,
        },
        'oth': {
            # 'cum_massflow': 90,
        },
    }
)
row1 = BladeRow(
    'Stator0',
    {
        'kin': {
            'alpha': Quantity(50, 'deg'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'chord': 0.2,
            'n_blades': 7,
            # 'height': 0.15,
            # 'solidity': 0.4,
        },
        'oth': {
            'mach': 0.4,
            'VmRatio': 0.9,
        },
    },
    shaft=static_shaft,
    extra_equations={
        # PercentageEntropyLoss(0.0): (0, 1),
        DentonProfileLoss(real_model): (0, 1),
        MeridionalVelocityRatio(): (0, 1),
    },
)

row2 = BladeRow(
    'Rotor0',
    {
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'chord': 0.2,
            'solidity': 0.4,
            # 'pitch': 0.15,
            # 'height': 0.2,
            # 'n_blades': 13,
        },
        'stc': {
            # 'p': 5e5,
        },
        'oth': {
            'workCoeff': 1.8,
            'VmRatio': 1.0,
        },
    },
    rotating_shaft,
    extra_equations={
        # DentonProfileLoss(real_model): (0, 1),
        PercentageEntropyLoss(0.0): (0, 1),
        WorkCoefficient(): (0, 1),
        MeridionalVelocityRatio(): (0, 1),
        HeightRatio(): (0, 1),
        FlowCoefficient(): 1,
    },
)
