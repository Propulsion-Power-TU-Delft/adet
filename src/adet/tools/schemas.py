from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Literal, Optional, Annotated
import logging
import warnings

logger = logging.getLogger(__name__)


class Inlet(BaseModel):
    """
    Defines the flow conditions at the inlet boundary of the turbomachine.
    Requires total pressure (Pt) and total temperature (Tt) as inputs.
    Flow angles alpha and phi default to 0.0 if not specified.
    Mass flow rate can be specified through any combination of three parameters:
    mass flow (mf), meridional velocity (Vm), radius (R), or height (H).

    :ivar Pt: Total pressure at inlet [Pa]
    :ivar Tt: Total temperature at inlet [K]
    :ivar alpha: Flow angle in meridional plane [rad]
    :ivar phi: Flow angle in tangential plane [rad]
    :ivar mf: Mass flow rate [kg/s]
    :ivar Vm: Meridional velocity [m/s]
    :ivar R: Radius [m]
    :ivar H: Height [m]
    """

    Pt: Annotated[float, Field(gt=0)]
    Tt: Annotated[float, Field(gt=0)]
    alpha: float = 0.0
    phi: float = 0.0
    mf: Annotated[Optional[float], Field(gt=0)] = None
    Vm: Annotated[Optional[float], Field(gt=0)] = None
    R: Annotated[Optional[float], Field(gt=0)] = None
    H: Annotated[Optional[float], Field(gt=0)] = None

    @model_validator(mode='after')
    @classmethod
    def check_mass_flow_input(cls, values):
        opt_variables = ['mf', 'Vm', 'R', 'H']
        defined_vars = [
            var for var in opt_variables if getattr(values, var) is not None
        ]
        if len(defined_vars) != 3:
            raise ValueError(
                f'Three of {opt_variables} must be defined at the inlet'
            )
        return values


class GlobalSettings(BaseModel):
    """
    Global configuration settings for the turbomachinery simulation.
    Defines the working fluid properties, computational domain discretization,
    and inlet boundary conditions.

    :ivar inlet: Inlet boundary conditions
    :ivar fluid: Working fluid name
    :ivar library: Thermodynamic property library to use
    :ivar n_span: Number of spanwise computational stations (must be odd)
    """

    inlet: Inlet
    fluid: str
    library: str
    n_span: int

    @field_validator('n_span', mode='after')
    @classmethod
    def check_n_span_odd(cls, value: int) -> int:
        if value % 2 == 0:
            warnings.warn(
                f'{value} spanwise stations set, rounding up to {value + 1}'
            )
            return value + 1
        else:
            return value


class GeometryDefaults(BaseModel):
    """
    Default geometry parameters applied to all components unless overridden.
    Specifies blade trailing edge thickness and tip clearance values that
    are used when not explicitly defined in component geometry.

    :ivar te_thickness: Default trailing edge thickness [m]
    :ivar tip_clearance: Default tip clearance gap [m]
    """

    te_thickness: float
    tip_clearance: float


class BladeRowConfig(BaseModel):
    """
    Configuration settings for blade row components (rotors or stators).
    Defines rotational speed either through shaft reference or direct omega value.
    Includes radial equilibrium model selection and component-specific constraints.

    :ivar name: Unique identifier for the blade row
    :ivar component_type: Type specification for blade row components
    :ivar shaft: Reference to defined shaft (optional)
    :ivar omega: Rotational speed in rad/s (optional)
    :ivar radial_equilibrium: Radial equilibrium model selection
    :ivar constraints: Dictionary of component-specific constraints
    """

    name: str
    component_type: Literal['blade_row', 'BladeRow', 'row', 'bladerow']
    shaft: Optional[str] = None
    omega: Optional[float] = None
    radial_equilibrium: str
    constraints: dict

    @model_validator(mode='after')
    @classmethod
    def validate_shaft_or_omega(cls, values):
        shaft = values.shaft
        omega = values.omega
        if (shaft is None and omega is None) or (
            shaft is not None and omega is not None
        ):
            raise ValueError(
                'Either "shaft" OR "omega" must be specified, but not both'
            )
        return values


class VoluteConfig(BaseModel):
    """
    Configuration settings for volute components.
    Specifies the component type and associated geometric constraints
    for volute modeling.

    :ivar name: Unique identifier for the volute
    :ivar component_type: Type specification for volute components
    :ivar constraints: Dictionary of component-specific constraints
    """

    name: str
    component_type: Literal['volute', 'Volute']
    constraints: dict


ComponentConfigs = Annotated[
    BladeRowConfig | VoluteConfig, Field(discriminator='component_type')
]


class ConfigFile(BaseModel):
    """
    Top-level configuration structure for the turbomachinery simulation.
    Contains global settings, constraints, shaft definitions, geometry defaults,
    and component specifications.

    :ivar settings: Global simulation settings
    :ivar global_constraints: Dictionary of constraints applied to all components
    :ivar shafts: Dictionary of shaft definitions and properties
    :ivar geometry_defaults: Default geometry parameters
    :ivar components: List of component configurations in flow path order
    """

    settings: GlobalSettings
    global_constraints: dict
    shafts: dict
    geometry_defaults: GeometryDefaults
    components: list[ComponentConfigs]

    @model_validator(mode='after')
    @classmethod
    def check_shaft_exists(cls, values):
        shafts = values.shafts
        components = values.components

        for comp in components:
            # Only check components that have a shaft attribute
            if hasattr(comp, 'shaft') and comp.shaft is not None:
                if comp.shaft not in shafts:
                    raise ValueError(
                        f"Shaft '{comp.shaft}' referenced by '{comp.name}' not found"
                    )
            return values
