"""
Sanity checks for the fluid models in adet.fluid
"""

from copy import deepcopy

import casadi as cs
import CoolProp as cp
import numpy as np
import pytest

from adet.fluid.algebraic_eos import IdealGasState
from adet.fluid.casadi_eos import CasadiEos
from adet.fluid.settings import FluidSettings
from adet.tools.coolprop_utils import DebugAbstractState
from adet.variables import ThermoVariables

GAMMA = 1.4
R_GAS = 287.0
VISCOSITY = 2e-5


class TestIdealGasState:
    def test_ps_update_matches_ideal_gas_relations(self):
        state = IdealGasState(gamma=GAMMA, sp_gas_constant=R_GAS, viscosity=VISCOSITY)
        p, s = 3e5, 1000.0
        state.update(cp.PSmass_INPUTS, p, s)

        cp_mass = R_GAS * GAMMA / (GAMMA - 1)
        T = np.exp((s + R_GAS * np.log(p)) / cp_mass)

        assert float(state.T()) == pytest.approx(T)
        assert float(state.rhomass()) == pytest.approx(p / (R_GAS * T))
        assert float(state.hmass()) == pytest.approx(cp_mass * T)

    def test_speed_of_sound_matches_ideal_gas_relation(self):
        state = IdealGasState(gamma=GAMMA, sp_gas_constant=R_GAS, viscosity=VISCOSITY)
        state.update(cp.PSmass_INPUTS, 3e5, 1000.0)

        T = float(state.T())
        assert float(state.speed_sound()) == pytest.approx(np.sqrt(GAMMA * R_GAS * T))


class TestCasadiEos:
    def test_values_match_coolprop(self):
        eos = DebugAbstractState('HEOS', 'Air')
        props = ['hmass', 'smass', 'rhomass']
        num_span = 7

        callback = CasadiEos(
            'PT_eos',
            eos,
            cp.PT_INPUTS,
            props,
            num_span,
            opts={'enable_fd': True, 'fd_method': 'forward'},
        )

        p_vals = cs.linspace(100e5, 600e5, num_span)  # Pressure [Pa]
        T_vals = cs.linspace(400, 600, num_span)  # Temperature [K]

        out = callback(p_vals, T_vals)

        expected = []
        for p, T in zip(np.array(p_vals).ravel(), np.array(T_vals).ravel()):
            eos.update(cp.PT_INPUTS, p, T)
            expected.append([eos.hmass(), eos.smass(), eos.rhomass()])
        expected = np.array(expected)

        for i in range(len(props)):
            got = np.array(out[i]).ravel()
            assert got == pytest.approx(expected[:, i], rel=1e-8)
