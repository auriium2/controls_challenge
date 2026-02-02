import os
import sys
import numpy as np
import pandas as pd
import pytest
import onnxruntime as ort

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator, LataccelTokenizer,
    CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, MAX_ACC_DELTA,
    STEER_RANGE, ACC_G, State,
)
from tinyphysics_eqx import create_model


class EQXModelWrapper:
    """Wraps the EQX model to present the same interface as TinyPhysicsModel.

    This lets us drop it into TinyPhysicsSimulator directly.
    """
    def __init__(self, eqx_model):
        self.eqx_model = eqx_model
        self.tokenizer = LataccelTokenizer()

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def predict(self, input_data, temperature=1.):
        states = jnp.array(input_data['states'])
        tokens = jnp.array(input_data['tokens'])
        logits = np.array(self.eqx_model(states, tokens))
        probs = self.softmax(logits / temperature, axis=-1)
        assert probs.shape[0] == 1
        assert probs.shape[2] == 1024
        sample = np.random.choice(probs.shape[2], p=probs[0, -1])
        return sample, probs[0, -1]

    def get_current_lataccel(self, sim_states, actions, past_preds):
        tokenized_actions = self.tokenizer.encode(past_preds)
        raw_states = [list(x) for x in sim_states]
        states = np.column_stack([actions, raw_states])
        input_data = {
            'states': np.expand_dims(states, axis=0).astype(np.float32),
            'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64),
        }
        sample, probs = self.predict(input_data, temperature=0.8)
        return self.tokenizer.decode(sample), probs


@pytest.fixture(scope="module")
def onnx_model():
    """Load ONNX TinyPhysicsModel once."""
    return TinyPhysicsModel('models/tinyphysics.onnx', debug=False)


@pytest.fixture(scope="module")
def eqx_wrapped_model():
    """Load EQX model wrapped to match TinyPhysicsModel interface."""
    eqx_model = create_model('models/tinyphysics.onnx')
    return EQXModelWrapper(eqx_model)


def get_data_files():
    """Get available test data files."""
    candidates = [0, 100, 500, 1000, 5000]
    available = []
    for idx in candidates:
        path = f'data/{idx:05d}.csv'
        if os.path.exists(path):
            available.append(path)
    return available


def run_simulator(model, data_path, controller_type='pid'):
    """Run TinyPhysicsSimulator exactly as eval.py does, return trajectory."""
    import importlib
    controller = importlib.import_module(f'controllers.{controller_type}').Controller()
    sim = TinyPhysicsSimulator(model, data_path, controller=controller, debug=False)
    cost = sim.rollout()
    return {
        'cost': cost,
        'current_lataccel_history': np.array(sim.current_lataccel_history),
        'target_lataccel_history': np.array(sim.target_lataccel_history),
        'action_history': np.array(sim.action_history),
    }


class TestEvalPipelineParity:
    """Test that EQX model produces identical results to ONNX in the full eval pipeline."""

    @pytest.mark.parametrize("data_path", get_data_files())
    def test_pid_rollout_identical(self, onnx_model, eqx_wrapped_model, data_path):
        """Run full eval pipeline with PID controller on both backends.

        Uses the same numpy RNG seed (set by TinyPhysicsSimulator.reset via
        md5 of data_path), so sampling is identical if logits match.
        """
        # Run ONNX
        onnx_result = run_simulator(onnx_model, data_path, 'pid')

        # Run EQX (same data_path -> same RNG seed in reset())
        eqx_result = run_simulator(eqx_wrapped_model, data_path, 'pid')

        # Compare trajectories
        onnx_lataccel = onnx_result['current_lataccel_history']
        eqx_lataccel = eqx_result['current_lataccel_history']

        assert len(onnx_lataccel) == len(eqx_lataccel), \
            f"Length mismatch: ONNX={len(onnx_lataccel)}, EQX={len(eqx_lataccel)}"

        # During warmup (steps 0 to CONTROL_START_IDX-1), both use ground truth
        # so they should be exactly equal
        warmup_diff = np.abs(onnx_lataccel[:CONTROL_START_IDX] - eqx_lataccel[:CONTROL_START_IDX]).max()
        assert warmup_diff == 0.0, f"Warmup mismatch: {warmup_diff}"

        # After control starts, small floating point differences in logits
        # could cause different samples, diverging the trajectories.
        # But if logits match closely (verified in test_weights.py), the
        # sampled tokens should be identical for the vast majority of steps.
        controlled = slice(CONTROL_START_IDX, COST_END_IDX)
        max_diff = np.abs(onnx_lataccel[controlled] - eqx_lataccel[controlled]).max()
        mean_diff = np.abs(onnx_lataccel[controlled] - eqx_lataccel[controlled]).mean()

        # With matching logits and same RNG, trajectories should be identical
        # Allow tiny tolerance for float32 accumulation differences
        assert max_diff < 1e-4, \
            f"Controlled region max diff: {max_diff:.6f} (mean: {mean_diff:.6f})"

    @pytest.mark.parametrize("data_path", get_data_files())
    def test_costs_identical(self, onnx_model, eqx_wrapped_model, data_path):
        """Verify that computed costs match between ONNX and EQX backends."""
        onnx_result = run_simulator(onnx_model, data_path, 'pid')
        eqx_result = run_simulator(eqx_wrapped_model, data_path, 'pid')

        for key in ['lataccel_cost', 'jerk_cost', 'total_cost']:
            onnx_val = onnx_result['cost'][key]
            eqx_val = eqx_result['cost'][key]
            assert abs(onnx_val - eqx_val) < 1e-3, \
                f"{key} mismatch: ONNX={onnx_val:.6f}, EQX={eqx_val:.6f}"

    @pytest.mark.parametrize("data_path", get_data_files())
    def test_actions_identical(self, onnx_model, eqx_wrapped_model, data_path):
        """Verify that PID controller produces same actions on both backends."""
        onnx_result = run_simulator(onnx_model, data_path, 'pid')
        eqx_result = run_simulator(eqx_wrapped_model, data_path, 'pid')

        onnx_actions = onnx_result['action_history']
        eqx_actions = eqx_result['action_history']

        assert len(onnx_actions) == len(eqx_actions)

        # During warmup, actions come from ground truth steer_command -> identical
        warmup_diff = np.abs(onnx_actions[:CONTROL_START_IDX] - eqx_actions[:CONTROL_START_IDX]).max()
        assert warmup_diff == 0.0, f"Warmup action mismatch: {warmup_diff}"

        # After control starts, PID actions depend on current_lataccel which
        # depends on model predictions. If predictions match, actions match.
        controlled_diff = np.abs(onnx_actions[CONTROL_START_IDX:] - eqx_actions[CONTROL_START_IDX:]).max()
        assert controlled_diff < 1e-4, \
            f"Controlled action max diff: {controlled_diff:.6f}"


class TestZeroControllerParity:
    """Test with zero controller (simplest case, no feedback effects)."""

    @pytest.mark.parametrize("data_path", get_data_files()[:2])
    def test_zero_controller_identical(self, onnx_model, eqx_wrapped_model, data_path):
        """Zero controller removes PID feedback, isolating model differences."""
        onnx_result = run_simulator(onnx_model, data_path, 'zero')
        eqx_result = run_simulator(eqx_wrapped_model, data_path, 'zero')

        onnx_lataccel = onnx_result['current_lataccel_history']
        eqx_lataccel = eqx_result['current_lataccel_history']

        controlled = slice(CONTROL_START_IDX, COST_END_IDX)
        max_diff = np.abs(onnx_lataccel[controlled] - eqx_lataccel[controlled]).max()
        assert max_diff < 1e-4, f"Zero controller max diff: {max_diff:.6f}"

        for key in ['lataccel_cost', 'jerk_cost', 'total_cost']:
            assert abs(onnx_result['cost'][key] - eqx_result['cost'][key]) < 1e-3
