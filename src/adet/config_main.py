"""Define some components"""

from pint import Quantity

# Equations
from adet.equations.definitions import HeightRatio, MeridionalVelocityRatio
from adet.equations.nondimensional import FlowCoefficient, WorkCoefficient
from adet.losses.profile import DentonProfileLoss, RectVelocityIncompressible
from adet.losses.basic import PercentageEntropyLoss

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
        'delta_smass_pct': 'J/(kg*K)',
        'deflection': 'rad',
        'percentage_loss': 'dimensionless',
        'workCoeff': 'dimensionless',
        'flowCoeff': 'dimensionless',
        'specificSpeed': 'dimensionless',
        'STratio': 'dimensionless',
        'VmRatio': 'dimensionless',
        'Vtmid': 'm/s',  # For vortex distributions
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
            'alpha': Quantity(0, 'deg'),
            # 'alpha': Quantity(30, 'deg'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'height': 0.2,
        },
        'tot': {
            'T': 700,
            'p': 6e5,  # impose at outlet
        },
        'oth': {
            'mach': 0.3,
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
            # Meridional
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            # Blade
            'chord': 0.15,
            'n_blades': 40,
            # 'solidity': 0.4,
        },
        'tot': {
            # 'p': 6e5, # Impose either here or at inlet
        },
        'oth': {
            'VmRatio': 0.95,
        },
    },
    shaft=static_shaft,
    extra_equations={
        MeridionalVelocityRatio(): (0, 1),
        # |> Losses
        PercentageEntropyLoss(0.0): (0, 1),
        # DentonProfileLoss(real_model): (0, 1),
    },
)

row2 = BladeRow(
    'Rotor0',
    {
        'kin': {
            # Repeated stage with axial discharge
            # 'alpha': Quantity(0, 'deg'),
        },
        'geo': {
            # Meridional
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            # Blade
            'chord': 0.15,
            # 'solidity': 0.4,
            'n_blades': 50,
            # NOTE: Insane sensitivity to number of blades????
            # Some tests:
            # - 40,42,43 do not converge
            # - 41,44,45,46,50 converge
        },
        'oth': {
            'workCoeff': 1.5,
            'heightRatio': 1.1,
            # 'VmRatio': 0.9,
        },
    },
    rotating_shaft,
    extra_equations={
        WorkCoefficient(): (0, 1),
        MeridionalVelocityRatio(): (0, 1),
        HeightRatio(): (0, 1),
        FlowCoefficient(): 1,
        # |> Losses
        DentonProfileLoss(real_model): (0, 1),
        # PercentageEntropyLoss(0.0): (0, 1),
    },
)
