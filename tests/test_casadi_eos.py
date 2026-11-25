"""
Structure and polymorphism checks for the
CasadiEos callback chain in adet.fluid.casadi_eos
"""

import copy

import casadi as cs
import CoolProp as cp
import numpy as np
import pytest

from adet.fluid.casadi_eos import (
    CasadiEos,
    CasadiEosHessian,
    CasadiEosJacobian,
    CasadiEosVaccarian,
)
from adet.tools.coolprop_utils import DebugAbstractState

NUM_SPAN = 4
PROPS = ['hmass', 'smass', 'rhomass']
N_IN = 2
N_OUT = len(PROPS)
N_IO = N_IN + N_OUT
OPTS = {'enable_fd': True, 'fd_method': 'forward'}

P_VALS = np.linspace(100e5, 600e5, NUM_SPAN)  # Pressure [Pa]
T_VALS = np.linspace(400, 600, NUM_SPAN)  # Temperature [K]


def _make_callback(props):
    eos = DebugAbstractState('HEOS', 'Air')
    return CasadiEos('PT_eos', eos, cp.PT_INPUTS, props, NUM_SPAN, opts=OPTS)


def _coolprop_value(eos, prop, p, T):
    eos.update(cp.PT_INPUTS, p, T)
    return getattr(eos, prop)()


def _fd_partial(eos, prop, p, T, wrt):
    """Central finite difference of a CoolProp property wrt input `wrt` (0=p, 1=T)"""
    if wrt == 0:
        h = 1e-6 * p
        up = _coolprop_value(eos, prop, p + h, T)
        down = _coolprop_value(eos, prop, p - h, T)
    else:
        h = 1e-6 * T
        up = _coolprop_value(eos, prop, p, T + h)
        down = _coolprop_value(eos, prop, p, T - h)
    return (up - down) / (2 * h)


@pytest.fixture
def callback():
    return _make_callback(PROPS)


class TestJacobianStructure:
    def test_io_counts(self, callback):
        jac = callback.jacobian()

        assert isinstance(jac, cs.Function)
        # Inputs: the eos inputs and the outputs of the forward call
        assert jac.n_in() == N_IO
        # One block per (output property, input variable) pair
        assert jac.n_out() == N_IN * N_OUT

    def test_blocks_are_diagonal_over_span(self, callback):
        jac = callback.jacobian()

        for k in range(jac.n_out()):
            sp = jac.sparsity_out(k)
            assert sp.shape == (NUM_SPAN, NUM_SPAN)
            assert sp.nnz() == NUM_SPAN
            assert sp.is_diag()

    def test_input_sparsity_is_dense_over_span(self, callback):
        jac = callback.jacobian()

        for i in range(N_IN):
            assert jac.sparsity_in(i).shape == (NUM_SPAN, 1)
            assert jac.sparsity_in(i).is_dense()

    def test_blocks_match_finite_difference(self, callback):
        p = cs.DM(P_VALS)
        T = cs.DM(T_VALS)
        outs = callback(p, T)

        blocks = callback.jacobian()(p, T, *outs)
        eos = DebugAbstractState('HEOS', 'Air')

        for prop_idx, prop in enumerate(PROPS):
            for input_idx in range(N_IN):
                block = np.array(blocks[prop_idx * N_IN + input_idx].full())

                # Off-diagonal entries couple different span points, so must vanish
                assert np.allclose(block - np.diag(np.diag(block)), 0.0)

                expected = [
                    _fd_partial(eos, prop, p_i, T_i, input_idx)
                    for p_i, T_i in zip(P_VALS, T_VALS)
                ]
                assert np.diag(block) == pytest.approx(expected, rel=1e-4)

    def _viscosity_blocks(self):
        # viscosity has no analytic derivative: the jacobian uses fwd_diff
        callback = _make_callback(['viscosity'])
        p = cs.DM(P_VALS)
        T = cs.DM(T_VALS)
        outs = (callback(p, T),)  # single output returns a bare DM
        return callback.jacobian()(p, T, *outs)

    def test_viscosity_jacobian_wrt_pressure_matches_finite_difference(self):
        blocks = self._viscosity_blocks()
        eos = DebugAbstractState('HEOS', 'Air')

        block = np.array(blocks[0].full())
        expected = [
            _fd_partial(eos, 'viscosity', p_i, T_i, 0)
            for p_i, T_i in zip(P_VALS, T_VALS)
        ]
        assert np.diag(block) == pytest.approx(expected, rel=1e-3)

    def test_viscosity_jacobian_wrt_temperature_matches_finite_difference(self):
        blocks = self._viscosity_blocks()
        eos = DebugAbstractState('HEOS', 'Air')

        block = np.array(blocks[1].full())
        expected = [
            _fd_partial(eos, 'viscosity', p_i, T_i, 1)
            for p_i, T_i in zip(P_VALS, T_VALS)
        ]
        assert np.diag(block) == pytest.approx(expected, rel=1e-3)


class TestHessianStructure:
    def test_io_counts(self, callback):
        hes = callback.jacobian().jacobian()

        # Inputs: eos inputs, outputs, and first derivative blocks
        assert hes.n_in() == N_IO + N_IN * N_OUT
        # One block per (output, input, input-or-output) triple
        assert hes.n_out() == N_IN * N_OUT * N_IO

    def test_non_hessiable_properties_have_zero_second_derivatives(self):
        # speed_sound is listed in NOT_HESSIABLE, so its block must be zero
        callback = _make_callback(['speed_sound', 'hmass'])
        p = cs.DM(P_VALS)
        T = cs.DM(T_VALS)
        outs = callback(p, T)
        jac_outs = callback.jacobian()(p, T, *outs)
        hes_outs = callback.jacobian().jacobian()(p, T, *outs, *jac_outs)

        # Hessian outputs are laid out as [prop][input][input-or-output];
        # for this callback each property block has N_IN * (N_IN + n_props) entries
        n_props = 2
        speed_sound_block_size = N_IN * (N_IN + n_props)
        for k in range(speed_sound_block_size):
            assert np.allclose(np.array(hes_outs[k].full()), 0.0)


class TestCallbackChain:
    def test_each_level_is_its_own_callback_class(self, callback):
        callback.jacobian().jacobian()

        assert isinstance(callback, CasadiEos)
        assert isinstance(callback.jac_callback, CasadiEosJacobian)
        assert isinstance(callback.jac_callback.hes_callback, CasadiEosHessian)
        for level in (
            callback,
            callback.jac_callback,
            callback.jac_callback.hes_callback,
        ):
            assert isinstance(level, cs.Callback)

    def test_third_level_is_zero_valued_vaccarian(self, callback):
        p = cs.DM(P_VALS)
        T = cs.DM(T_VALS)
        outs = callback(p, T)
        jac_outs = callback.jacobian()(p, T, *outs)
        hes_outs = callback.jacobian().jacobian()(p, T, *outs, *jac_outs)

        vac = callback.jacobian().jacobian().jacobian()
        vac_outs = vac(p, T, *outs, *jac_outs, *hes_outs)

        assert isinstance(
            callback.jac_callback.hes_callback.vac_callback, CasadiEosVaccarian
        )
        assert callback.jac_callback.hes_callback.vac_callback.has_jacobian() is False
        for out in vac_outs if isinstance(vac_outs, tuple) else (vac_outs,):
            assert np.allclose(np.array(out.full()), 0.0)

    def test_call_dm_and_mx_inputs_agree(self, callback):
        p = cs.DM(P_VALS)
        T = cs.DM(T_VALS)
        numeric = callback(p, T)

        v0 = cs.MX.sym('p', NUM_SPAN)
        v1 = cs.MX.sym('T', NUM_SPAN)
        symbolic = cs.Function('f', [v0, v1], list(callback(v0, v1)))
        evaluated = symbolic(p, T)

        for num, sym in zip(numeric, evaluated):
            assert np.allclose(np.array(num.full()), np.array(sym.full()), rtol=1e-10)

    def test_deepcopy_gives_equivalent_callback(self, callback):
        clone = copy.deepcopy(callback)

        assert isinstance(clone, CasadiEos)
        assert clone is not callback
        assert clone._output_props == callback._output_props
        assert clone._input_pair == callback._input_pair
        assert clone._num_span == callback._num_span

        p = cs.DM(P_VALS)
        T = cs.DM(T_VALS)
        for a, b in zip(callback(p, T), clone(p, T)):
            assert np.allclose(np.array(a.full()), np.array(b.full()))
