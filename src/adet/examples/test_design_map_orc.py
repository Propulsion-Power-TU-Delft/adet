"""Functional tests for the ORC axial turbine design map.

Each test builds its own independent system and solves a single (phi, psi) point
to keep runtime reasonable.  Run before and after changes to design_map_orc.py:

  Before changes: test_isentropic_initial passes, test_loss_initial passes
                  (the loss logic lives here, not in the script under test)
  After changes:  both tests still pass — they validate shared infrastructure
"""
from copy import deepcopy
from typing import Type

import CoolProp as cp
import numpy as np
import pytest
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, ComponentNetwork, Inlet, Shaft
from adet.equations.base_equation import EquationBase, LossApplier
from adet.equations.definitions import (
    BoundaryLayerRatios,
    ClearanceByHeight,
    IsentropicProperties,
    RepeatedStage,
)
from adet.equations.geometrical import MinimalCamberLine
from adet.equations.nondimensional import (
    FlowCoefficient,
    StaticTotalDegreeOfReaction,
    TotalStaticLoadingCoefficient,
    TotalTotalExpansionEfficiency,
    WorkCoefficient,
)
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.losses.leakage import DentonLeakageLoss
from adet.losses.mixing import SieverdingBasePressure
from adet.losses.profile import DentonProfileLoss
from adet.losses.secondary import SecondaryBSM
from adet.registries import DefaultUnitsRegistry, GuessRegistry, VariableBoundsRegistry
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState

# Reference design point (same as design_map_orc.py)
_PHI_0 = 0.4
_PSI_0 = 3.0

EXTRA_EQUATIONS: dict[Type[EquationBase], int | tuple[int, ...]] = {
    ClearanceByHeight: 1,
    IsentropicProperties: (0, 1),
    BoundaryLayerRatios: 1,
    SieverdingBasePressure: (0, 1),
    SecondaryBSM: (0, 1),
    DentonProfileLoss: (0, 1),
    DentonLeakageLoss: (0, 1),
}


class LossMatcher(LossApplier):
    def __init__(self, tip_gap: bool, scaling_factor=None):
        super().__init__(scaling_factor)
        self.tip_gap = tip_gap

    def residual(
        self,
        stc_smass0,
        stc_smass1,
        oth_delta_smass_leakage1,
        oth_delta_smass_profile1,
        oth_delta_smass_secondary1,
    ):
        if self.tip_gap:
            return stc_smass1 - (
                stc_smass0
                + oth_delta_smass_leakage1
                + oth_delta_smass_profile1
                + oth_delta_smass_secondary1
            )
        return stc_smass1 - (
            stc_smass0 + oth_delta_smass_profile1 + oth_delta_smass_secondary1
        )


def _setup_orc_system():
    """Build and return the isentropic ORC single-stage network at the reference point."""
    abs_state = DebugAbstractState('REFPROP', 'MM')
    abs_state.debug_print = False

    real_model = ExternalFluidModel(abs_state)
    INLET_PRESSURE = 2.071 * abs_state.p_critical()
    INLET_TEMPERATURE = 1.052 * abs_state.T_critical()
    abs_state.update(cp.PT_INPUTS, INLET_PRESSURE, INLET_TEMPERATURE)

    fluid_settings = FluidSettings(
        model=real_model,
        update_variables=('p', 'hmass', 'T'),
        update_length=2,
    )

    _defreg = DefaultUnitsRegistry()
    _defreg.from_dict(
        {
            'xi_by_camb_.*': 'dimensionless',
            'Cd_prof': 'dimensionless',
            'k_prof': 'dimensionless',
            'Y_.*': 'dimensionless',
        }
    )
    _guess_reg = GuessRegistry()
    _guess_reg.reset()
    _guess_reg.from_dict(
        {
            'hdropCoeff': -0.8,
            'workCoeff': -0.8,
            'reactDegree_ts': 0.5,
            'p': abs_state.p(),
            'T': abs_state.T(),
            'hmass': abs_state.hmass(),
            'smass': abs_state.smass(),
            'rhomass': abs_state.rhomass(),
            'k_prof': 0.3,
        }
    )
    _bounds_reg = VariableBoundsRegistry()
    _bounds_reg.reset()
    _bounds_reg.from_dict({'hdropCoeff': (-8.0, -0.4), 'U': (0.0, 200.0)})
    _bounds_reg.from_dict(
        {
            'p': (abs_state.p_critical(), INLET_PRESSURE),
            'T': (abs_state.T_critical(), INLET_TEMPERATURE),
            'hmass': (abs_state.hmass() - 2 * 60**2, 1.2 * abs_state.hmass()),
        }
    )

    casing = Shaft(0.0, is_constrained=True)
    shaft = Shaft(-1, is_constrained=False)
    initial_loss = PercentageEntropyLoss(0.0)

    inlet = Inlet(
        {
            'oth': {'cum_massflow': 1},
            'kin': {'mermach': 0.1},
            'geo': {
                'meridional_angle': Quantity(0, 'deg'),
                'hubtipRatio': 0.7,
            },
            'tot': {
                'p': abs_state.p(),
                'hmass': abs_state.hmass(),
            },
        }
    )

    stator = BladeRow(
        name='Stator',
        shaft=casing,
        row_type='stator',
        in_constraints={'geo': {'thick_by_pitch': 0.04}},
        out_constraints={
            'geo': {
                'meridional_angle': Quantity(0, 'deg'),
                'aspRatio': 2,
                'num_blades': 25,
                'thick_by_pitch': 0.02,
                'clearance_by_height': 0.01,
            },
            'oth': {
                'mom_by_bld': 0.075,
                'disp_by_mom': 2,
                'disp_by_hgt': 0.05,
                'Cd_profile': 0.002,
                'xi_by_camb_len_A': 0.375,
                'xi_by_camb_len_B': 0.675,
                'dischCoeff': 0.35,
            },
        },
        extra_equations={
            MinimalCamberLine(): (0, 1),
            ZeroDeviation(): 0,
            ZeroDeviation(): 1,
            initial_loss: (0, 1),
        },
        constant_variables=['geo_rr_midspan'],
    )

    rotor = deepcopy(stator)
    rotor.shaft = shaft
    rotor.row_type = 'rotor'
    rotor._equations.update(
        {
            WorkCoefficient(): (0, 1),
            FlowCoefficient(): (0, 1),
        }
    )

    ntw = ComponentNetwork(
        fluid_settings, inlet, CasadiSystem(num_span=1), components=[stator, rotor]
    )
    final_node = ntw.num_components * 2 - 1

    ntw.system.boundary_conditions[final_node]['oth']['flowCoeff'] = _PHI_0
    ntw.system.boundary_conditions[final_node]['oth']['reactDegree_ts'] = 0.3
    ntw.system.boundary_conditions[final_node]['oth']['ts_loadCoeff'] = _PSI_0

    ntw.system.add_spanwise_constants('geo_hh0', 'geo_chord_ax1', 'geo_chord_ax3')
    ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, final_node))
    ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
    ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, final_node))
    ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))

    ntw.system.build(True)

    return ntw, final_node, initial_loss, stator, rotor


def _solve_isentropic(ntw):
    rootfinder = ntw.system.make_rootfinder(
        'ipopt', opts={'error_on_fail': False, 'ipopt.max_iter': 1000}
    )
    x0 = ntw.system.get_scaled_guess()
    kn = ntw.system.get_scaled_constraints()
    bnd = ntw.system.get_arguments_bounds()
    return solve_root_problem(
        rootfinder, x0, kn, bnd, suppress_output=True, perturbate_guess=False
    )


# ================================================
def test_isentropic_initial():
    """Isentropic solve at reference point produces a finite work coefficient."""
    ntw, final_node, *_ = _setup_orc_system()
    solution = _solve_isentropic(ntw)

    work_coeff_key = f'oth_workCoeff{final_node}'
    sol_dict = ntw.system.solution_to_dict(solution)

    assert work_coeff_key in sol_dict, (
        f'work_coeff key {work_coeff_key!r} missing from isentropic solution'
    )
    wc = float(sol_dict[work_coeff_key][0])
    assert np.isfinite(wc), f'work_coeff is not finite: {wc}'
    assert wc < 0, f'Expected negative work coefficient for a turbine, got {wc:.4f}'


def test_loss_initial():
    """Loss solve at reference point produces an eta_tt in (0.3, 1.0)."""
    ntw, final_node, initial_loss, stator, rotor = _setup_orc_system()
    is_solution = _solve_isentropic(ntw)
    sol_dict_is = ntw.system.solution_to_dict(is_solution)

    # Transition to losses
    ntw.system.remove_equation_type(LossApplier)
    rotor.remove_equation(initial_loss.__class__, (0, 1))
    stator.remove_equation(initial_loss.__class__, (0, 1))
    for eq, pos in EXTRA_EQUATIONS.items():
        rotor.add_equation(eq(), pos)
        stator.add_equation(eq(), pos)
    rotor.add_equation(LossMatcher(tip_gap=True), (0, 1))
    stator.add_equation(LossMatcher(tip_gap=False), (0, 1))
    ntw.build()

    x0_loss = ntw.system.get_scaled_guess(sol_dict_is)
    kn_loss = ntw.system.get_scaled_constraints()
    bnd_loss = ntw.system.get_arguments_bounds()

    rootfinder_loss = ntw.system.make_rootfinder(
        'ipopt', opts={'error_on_fail': False, 'ipopt.max_iter': 1000}
    )
    solution_loss = solve_root_problem(
        rootfinder_loss,
        x0_loss,
        kn_loss,
        bnd_loss,
        suppress_output=True,
        perturbate_guess=False,
    )

    eta_tt_key = f'oth_eta_tt{final_node}'
    sol_dict = ntw.system.solution_to_dict(solution_loss)

    assert eta_tt_key in sol_dict, (
        f'eta_tt key {eta_tt_key!r} missing from loss solution'
    )
    eta_tt = float(sol_dict[eta_tt_key][0])
    assert np.isfinite(eta_tt), f'eta_tt is not finite: {eta_tt}'
    assert 0.3 < eta_tt < 1.0, f'eta_tt={eta_tt:.4f} outside expected range (0.3, 1.0)'
