from enum import Enum
from typing import Union

import CoolProp as cp
from numpy.typing import NDArray


class CoolProperties(Enum):
    Press = 'p'
    Temp = 'T'
    Hmass = 'hmass'
    Umass = 'umass'
    Smass = 'smass'
    Dmass = 'rhomass'
    Cpmass = 'cpmass'
    Cvmass = 'cvmass'
    Pcrit = 'p_critical'
    Tcrit = 'T_critical'
    Quality = 'Q'
    SpeedSound = 'speed_sound'
    Viscosity = 'viscosity'
    GasConstant = 'gas_constant'
    MolarMass = 'molar_mass'


COOLPROP_NAMES_MAP = {
    CoolProperties.Press.value: 'P',
    CoolProperties.Temp.value: 'T',
    CoolProperties.Quality.value: 'Q',
    CoolProperties.Hmass.value: 'Hmass',
    CoolProperties.Umass.value: 'Umass',
    CoolProperties.Smass.value: 'Smass',
    CoolProperties.Dmass.value: 'Dmass',
    CoolProperties.Cpmass.value: 'Cpmass',
    CoolProperties.Cvmass.value: 'Cvmass',
    CoolProperties.Pcrit.value: 'P_critical',
}

INVERSE_CP_NAMES_MAP = {v: k for k, v in COOLPROP_NAMES_MAP.items()}

_PAIR_SUFFIX = '_INPUTS'
COOLPROP_PAIRS: dict[int, str] = {
    getattr(cp, key): key[: -len(_PAIR_SUFFIX)]
    for key in dir(cp)
    if key.endswith(_PAIR_SUFFIX)
}

AdetArray = Union[
    NDArray,
    list[float],
    list[int],
    tuple[float, ...],
    tuple[int, ...],
    float,
    int,
]

IPOPT_DEFAULTS = {
    'error_on_fail': True,
    # Reasonable defaults for IPOPT, overwritten by user
    'ipopt.print_level': 3,
    'ipopt.max_iter': 6000,
    'ipopt.max_wall_time': 60,
    'ipopt.tol': 1e-8,
    'ipopt.acceptable_constr_viol_tol': 1e-10,
    'ipopt.bound_frac': 0.1,  # Relative initial push < 0.5
    'ipopt.mu_init': 0.3,  # Initial barrier param
    'ipopt.mu_strategy': 'adaptive',
    'ipopt.linear_solver': 'spral',
    # Lower = stricter restoration (def = 100 * tol)
    'ipopt.resto_failure_feasibility_threshold': 1e-7,
    'ipopt.expect_infeasible_problem': 'yes',
    'ipopt.hessian_approximation': 'limited-memory',  # Less updates
}
