import functools
import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
import equinox.nn as nn
import numpy as np
import onnx
import onnx.numpy_helper
from typing import Dict, Tuple
from functools import partial
from jaxopt.projection import projection_simplex
from jaxtyping import Float, Array
from common import BINS, TEMPERATURE, MAX_ACC_DELTA, VELOCITY_CLIP, decode, encode

class TransformerBlock(eqx.Module):
    """Single transformer block with pre-norm architecture."""

    ln1: nn.LayerNorm
    ln2: nn.LayerNorm
    attn_qkv: nn.Linear
    attn_proj: nn.Linear
    mlp_fc: nn.Linear
    mlp_proj: nn.Linear
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, key):
        keys = jax.random.split(key, 4)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, use_bias=False, key=keys[0])
        self.attn_proj = nn.Linear(d_model, d_model, use_bias=False, key=keys[1])
        self.mlp_fc = nn.Linear(d_model, d_ff, use_bias=False, key=keys[2])
        self.mlp_proj = nn.Linear(d_ff, d_model, use_bias=False, key=keys[3])

    def __call__(self, x, mask):
        B, T, C = x.shape
        scale = self.head_dim ** -0.5

        def ln(x, weight, bias, eps=1e-5):
            """Layer norm on last dim, works with any batch dims."""
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + eps) * weight + bias

        # Pre-norm attention - direct matmul instead of vmap
        h = ln(x, self.ln1.weight, self.ln1.bias)
        qkv = h @ self.attn_qkv.weight.T
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = jnp.where(mask, attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape back and project
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = out @ self.attn_proj.weight.T
        x = x + out

        # Pre-norm MLP
        h = ln(x, self.ln2.weight, self.ln2.bias)
        h = h @ self.mlp_fc.weight.T
        h = jax.nn.gelu(h, approximate=True)
        h = h @ self.mlp_proj.weight.T
        x = x + h

        return x
class EQXPhysicsModel(eqx.Module):
    """TinyPhysics GPT-style model."""

    state_proj: nn.Linear
    token_embed: nn.Embedding
    pos_embed: jax.Array
    blocks: list
    ln_f: nn.LayerNorm
    lm_head: nn.Linear
    context_len: int = eqx.field(static=True)

    def __init__(self, n_layers: int = 4, n_heads: int = 4, d_model: int = 128, d_ff: int = 512, context_len: int = 20, vocab_size: int = 1024, state_dim: int = 4, *,  key):
        keys = jax.random.split(key, n_layers + 4)
        self.context_len = context_len

        # Input projections
        self.state_proj = nn.Linear(state_dim, d_model // 2, key=keys[0])
        self.token_embed = nn.Embedding(vocab_size, d_model // 2, key=keys[1])
        self.pos_embed = jax.random.normal(keys[2], (context_len, d_model)) * 0.02

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, key=keys[3 + i])
            for i in range(n_layers)
        ]

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, use_bias=False, key=keys[-1])
    def __call__(self, states, tokens):
        """
        Forward pass.

        Args:
            states: [batch, seq, 4] - (lataccel, action, roll, v)
            tokens: [batch, seq] - tokenized lataccel history

        Returns:
            logits: [batch, seq, vocab_size]
        """
        B, T, _ = states.shape

        def ln(x, weight, bias, eps=1e-5):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + eps) * weight + bias

        # Embeddings - direct matmul instead of vmap
        state_emb = states @ self.state_proj.weight.T + self.state_proj.bias
        token_emb = self.token_embed.weight[tokens]  # [B, T, 64]

        # Concatenate and add position
        x = jnp.concatenate([state_emb, token_emb], axis=-1)
        x = x + self.pos_embed[:T]

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output
        x = ln(x, self.ln_f.weight, self.ln_f.bias)
        logits = x @ self.lm_head.weight.T

        return logits
def load_weights_from_onnx(model: EQXPhysicsModel, onnx_path: str) -> EQXPhysicsModel:
    """Load weights from ONNX file into Equinox model."""
    onnx_model = onnx.load(onnx_path)

    weights = {}
    for init in onnx_model.graph.initializer:
        weights[init.name] = onnx.numpy_helper.to_array(init).copy()

    # State projection
    model = eqx.tree_at(
        lambda m: m.state_proj.weight,
        model,
        jnp.array(weights['onnx::MatMul_540'].T)  # [64, 4]
    )
    model = eqx.tree_at(
        lambda m: m.state_proj.bias,
        model,
        jnp.array(weights['transformer.wt_embedding.bias'])
    )

    # Token embedding
    model = eqx.tree_at(
        lambda m: m.token_embed.weight,
        model,
        jnp.array(weights['transformer.wt2_embedding.weight'])
    )

    # Position embedding
    model = eqx.tree_at(
        lambda m: m.pos_embed,
        model,
        jnp.array(weights['transformer.wp_embedding.weight'])
    )

    # Transformer blocks
    block_weights = [
        ('onnx::MatMul_542', 'onnx::MatMul_546', 'onnx::MatMul_547', 'onnx::MatMul_548'),
        ('onnx::MatMul_549', 'onnx::MatMul_553', 'onnx::MatMul_554', 'onnx::MatMul_555'),
        ('onnx::MatMul_556', 'onnx::MatMul_560', 'onnx::MatMul_561', 'onnx::MatMul_562'),
        ('onnx::MatMul_563', 'onnx::MatMul_567', 'onnx::MatMul_568', 'onnx::MatMul_569'),
    ]

    for i, (c_attn, c_proj, c_fc, mlp_proj) in enumerate(block_weights):
        # Layer norms
        model = eqx.tree_at(
            lambda m, i=i: m.blocks[i].ln1.weight,
            model,
            jnp.array(weights[f'transformer.h.{i}.attn.layer_norm.weight'])
        )
        model = eqx.tree_at(
            lambda m, i=i: m.blocks[i].ln1.bias,
            model,
            jnp.array(weights[f'transformer.h.{i}.attn.layer_norm.bias'])
        )
        model = eqx.tree_at(
            lambda m, i=i: m.blocks[i].ln2.weight,
            model,
            jnp.array(weights[f'transformer.h.{i}.mlp.layer_norm.weight'])
        )
        model = eqx.tree_at(
            lambda m, i=i: m.blocks[i].ln2.bias,
            model,
            jnp.array(weights[f'transformer.h.{i}.mlp.layer_norm.bias'])
        )

        # Attention weights (transposed for Linear)
        model = eqx.tree_at(
            lambda m, i=i: m.blocks[i].attn_qkv.weight,
            model,
            jnp.array(weights[c_attn].T)
        )
        model = eqx.tree_at(
            lambda m, i=i: m.blocks[i].attn_proj.weight,
            model,
            jnp.array(weights[c_proj].T)
        )

        # MLP weights
        model = eqx.tree_at(
            lambda m, i=i: m.blocks[i].mlp_fc.weight,
            model,
            jnp.array(weights[c_fc].T)
        )
        model = eqx.tree_at(
            lambda m, i=i: m.blocks[i].mlp_proj.weight,
            model,
            jnp.array(weights[mlp_proj].T)
        )

    # Final layer norm
    model = eqx.tree_at(
        lambda m: m.ln_f.weight,
        model,
        jnp.array(weights['transformer.layer_norm_f.weight'])
    )
    model = eqx.tree_at(
        lambda m: m.ln_f.bias,
        model,
        jnp.array(weights['transformer.layer_norm_f.bias'])
    )

    # LM head
    model = eqx.tree_at(lambda m: m.lm_head.weight,model,jnp.array(weights['onnx::MatMul_570'].T) )

    return model
def create_model(onnx_path: str) -> EQXPhysicsModel:
    """Create model and load weights from ONNX."""
    key = jax.random.PRNGKey(0)
    model = EQXPhysicsModel(key=key)
    model = load_weights_from_onnx(model, onnx_path)
    return model



#
type PredictorState = Tuple[Float[Array, 20], Float[Array, 20], Float[Array, 20], Float[Array, 1]]
type PredictorExo = Float[Array, 4]

def predict_raw(model: EQXPhysicsModel, state: PredictorState):
    action_hist, lataccel_hist, exo_hist, current_lataccel = state

    pass


def make_simulation_step(model):
    def simulation_step(carry, inputs):
        """Single simulation step.

        States format (from tinyphysics.py):
            states = np.column_stack([actions, raw_states])
            where raw_states = [roll_lataccel, v_ego, a_ego]

        So states[t] = [action[t], roll[t], v[t], a[t]]

        Tokens are tokenized past lataccel predictions.

        Order of operations (matching original TinyPhysicsSimulator):
        1. Use current action_hist, exo_hist, lataccel_hist for forward pass
        2. Get prediction
        3. Update histories with new values for next step
        """
        action_hist, lataccel_hist, exo_hist, current_lataccel = carry
        exo_row, action = inputs  # exo_row: [batch, 4] = (roll, v, a, target), action: [batch]

        # Build states [batch, 20, 4]: (action, roll, v, a)
        # Use histories (before updating with new values)
        states = jnp.concatenate([
            action_hist[:, :, None],  # [batch, 20, 1]
            exo_hist,                  # [batch, 20, 3] = roll, v, a
        ], axis=-1)

        # Tokenize lataccel history (past predictions)
        tokens = encode(lataccel_hist)

        # Forward pass
        logits = model(states, tokens)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        next_lataccel = decode(next_token)

        # Rate limit
        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        # update histories for next step
        action_hist = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
        exo_hist = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)  # Only roll, v, a (not target)
        lataccel_hist = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)

        new_carry = (action_hist, lataccel_hist, exo_hist, next_lataccel)
        output = jnp.stack([next_lataccel, action, exo_row[:, 0], exo_row[:, 1], exo_row[:, 2], exo_row[:, 3]], axis=-1)
        return new_carry, output

    return simulation_step


def run_simulation(model, init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, actions):
    """
    Run batched simulation using lax.scan.

    Args:
        model: TinyPhysicsModel
        init_action_hist: [batch, 20] - initial action history
        init_lataccel_hist: [batch, 20] - initial lataccel history
        init_exo_hist: [batch, 20, 3] - initial exo history (roll, v, a)
        exo_data: [n_steps, batch, 4] (roll, v, a, target)
        actions: [n_steps, batch]

    Returns:
        outputs: [n_steps, batch, 6] (lataccel, action, roll, v, a, target)
    """
    init_carry = (init_action_hist, init_lataccel_hist, init_exo_hist, init_lataccel_hist[:, -1])
    step_fn = make_simulation_step(model)
    _, outputs = lax.scan(step_fn, init_carry, (exo_data, actions))
    return outputs


# ============================================================================
# PID CONTROLLER SIMULATION
# ============================================================================

def make_pid_simulation_step(model, p=0.195, i=0.100, d=-0.053):
    """Create a simulation step function with PID controller."""

    def simulation_step(carry, inputs):
        """Single simulation step with PID control."""
        action_hist, lataccel_hist, exo_hist, current_lataccel, error_integral, prev_error = carry
        exo_row = inputs  # exo_row: [batch, 4] = (roll, v, a, target)

        target = exo_row[:, 3]  # target lataccel

        # PID control
        error = target - current_lataccel
        error_integral_new = error_integral + error
        error_diff = error - prev_error

        action = p * error + i * error_integral_new + d * error_diff
        action = jnp.clip(action, -2, 2)

        # Build states [batch, 20, 4]: (action, roll, v, a)
        states = jnp.concatenate([
            action_hist[:, :, None],
            exo_hist,
        ], axis=-1)

        # Tokenize lataccel history
        tokens = encode(lataccel_hist)

        # Forward pass
        logits = model(states, tokens)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        next_lataccel = decode(next_token)

        # Rate limit
        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        # Update histories
        action_hist = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
        exo_hist = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)
        lataccel_hist = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)

        new_carry = (action_hist, lataccel_hist, exo_hist, next_lataccel, error_integral_new, error)
        output = jnp.stack([next_lataccel, action, exo_row[:, 0], exo_row[:, 1], exo_row[:, 2], exo_row[:, 3]], axis=-1)
        return new_carry, output

    return simulation_step


def run_simulation_pid(model, init_action_hist, init_lataccel_hist, init_exo_hist, exo_data,
                       p=0.195, i=0.100, d=-0.053):
    """
    Run batched simulation with PID controller.

    Args:
        model: TinyPhysicsModel
        init_action_hist: [batch, 20] - initial action history
        init_lataccel_hist: [batch, 20] - initial lataccel history
        init_exo_hist: [batch, 20, 3] - initial exo history (roll, v, a)
        exo_data: [n_steps, batch, 4] (roll, v, a, target)
        p, i, d: PID gains

    Returns:
        outputs: [n_steps, batch, 6] (lataccel, action, roll, v, a, target)
    """
    batch_size = init_action_hist.shape[0]
    init_error_integral = jnp.zeros(batch_size)
    init_prev_error = jnp.zeros(batch_size)

    init_carry = (
        init_action_hist,
        init_lataccel_hist,
        init_exo_hist,
        init_lataccel_hist[:, -1],
        init_error_integral,
        init_prev_error
    )
    step_fn = make_pid_simulation_step(model, p, i, d)
    _, outputs = lax.scan(step_fn, init_carry, exo_data)
    return outputs


def make_noisy_pid_simulation_step(model, p=0.195, i=0.100, d=-0.053, noise_std=0.3):
    """Create a simulation step function with noisy PID controller."""

    def simulation_step(carry, inputs):
        """Single simulation step with noisy PID control."""
        action_hist, lataccel_hist, exo_hist, current_lataccel, error_integral, prev_error = carry
        exo_row, noise = inputs  # exo_row: [batch, 4], noise: [batch]

        target = exo_row[:, 3]

        # PID control with noise
        error = target - current_lataccel
        error_integral_new = error_integral + error
        error_diff = error - prev_error

        action = p * error + i * error_integral_new + d * error_diff
        action = action + noise  # Add exploration noise
        action = jnp.clip(action, -2, 2)

        # Build states
        states = jnp.concatenate([
            action_hist[:, :, None],
            exo_hist,
        ], axis=-1)

        tokens = encode(lataccel_hist)

        logits = model(states, tokens)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        next_lataccel = decode(next_token)

        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        action_hist = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
        exo_hist = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)
        lataccel_hist = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)

        new_carry = (action_hist, lataccel_hist, exo_hist, next_lataccel, error_integral_new, error)
        output = jnp.stack([next_lataccel, action, exo_row[:, 0], exo_row[:, 1], exo_row[:, 2], exo_row[:, 3]], axis=-1)
        return new_carry, output

    return simulation_step


def run_simulation_noisy_pid(model, init_action_hist, init_lataccel_hist, init_exo_hist, exo_data,
                              key, p=0.195, i=0.100, d=-0.053, noise_std=0.3):
    """
    Run batched simulation with noisy PID controller for exploration.

    Args:
        model: TinyPhysicsModel
        init_action_hist: [batch, 20]
        init_lataccel_hist: [batch, 20]
        init_exo_hist: [batch, 20, 3]
        exo_data: [n_steps, batch, 4] (roll, v, a, target)
        key: JAX random key
        p, i, d: PID gains
        noise_std: standard deviation of exploration noise

    Returns:
        outputs: [n_steps, batch, 6] (lataccel, action, roll, v, a, target)
    """
    batch_size = init_action_hist.shape[0]
    n_steps = exo_data.shape[0]

    init_error_integral = jnp.zeros(batch_size)
    init_prev_error = jnp.zeros(batch_size)

    # Pre-generate noise for all steps
    noise = jax.random.normal(key, (n_steps, batch_size)) * noise_std

    init_carry = (
        init_action_hist,
        init_lataccel_hist,
        init_exo_hist,
        init_lataccel_hist[:, -1],
        init_error_integral,
        init_prev_error,
    )
    step_fn = make_noisy_pid_simulation_step(model, p, i, d, noise_std)
    _, outputs = lax.scan(step_fn, init_carry, (exo_data, noise))
    return outputs


# ============================================================================
# DIFFERENTIABLE SIMULATION (for training surrogate models)
# ============================================================================


def make_differentiable_step(model, temperature=0.1):
    """Create a fully differentiable simulation step.

    Uses soft encoding/decoding so gradients flow through the entire
    recurrence, including the lataccel history.
    """

    def simulation_step(carry, inputs):
        """Fully differentiable simulation step."""
        action_hist, lataccel_hist, exo_hist, current_lataccel = carry
        exo_row, action = inputs  # exo_row: [batch, 4], action: [batch]

        # Build states [batch, 20, 4]: (action, roll, v, a)
        states = jnp.concatenate([
            action_hist[:, :, None],
            exo_hist,
        ], axis=-1)

        # Forward pass with SOFT token embeddings (differentiable)
        logits = model_forward_soft(model, states, lataccel_hist, temperature=temperature)

        # DIFFERENTIABLE: use soft decode
        next_lataccel = soft_decode(logits[:, -1, :], temperature=temperature)

        # Rate limit
        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        # Update histories
        action_hist = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
        exo_hist = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)
        lataccel_hist = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)

        new_carry = (action_hist, lataccel_hist, exo_hist, next_lataccel)
        return new_carry, next_lataccel

    return simulation_step


def run_simulation_differentiable(model, init_action_hist, init_lataccel_hist, init_exo_hist,
                                   exo_data, actions, temperature=0.1):
    """
    Run differentiable simulation - gradients flow through the entire rollout.

    Args:
        model: TinyPhysicsModel (frozen weights)
        init_action_hist: [batch, 20]
        init_lataccel_hist: [batch, 20]
        init_exo_hist: [batch, 20, 3]
        exo_data: [n_steps, batch, 4] (roll, v, a, target)
        actions: [n_steps, batch]
        temperature: softmax temperature for soft decoding

    Returns:
        lataccels: [n_steps, batch] - predicted lateral accelerations
    """
    init_carry = (init_action_hist, init_lataccel_hist, init_exo_hist, init_lataccel_hist[:, -1])
    step_fn = make_differentiable_step(model, temperature=temperature)
    _, lataccels = lax.scan(step_fn, init_carry, (exo_data, actions))
    return lataccels


# ============================================================================
# STRAIGHT-THROUGH ESTIMATOR SIMULATION
# Hard forward pass, soft backward pass for gradients
# Uses custom_vjp to ensure forward pass is EXACTLY the same as hard simulation
# ============================================================================

def _decode_fwd(logits, temperature):
    """Forward pass: exact hard decode."""
    token = jnp.argmax(logits, axis=-1)
    hard_value = BINS[jnp.clip(token, 0, 1023)]
    # Store logits for backward pass
    return hard_value, logits

def _decode_bwd(temperature, res, g):
    """Backward pass: use soft decode gradient.

    Note: When using nondiff_argnums, the non-diff args come first in bwd.
    """
    logits = res
    # Compute gradient of soft_decode w.r.t. logits
    def soft_decode_for_grad(logits):
        probs = jax.nn.softmax(logits / temperature, axis=-1)
        return jnp.sum(probs * BINS, axis=-1)

    _, vjp_fn = jax.vjp(soft_decode_for_grad, logits)
    logits_grad, = vjp_fn(g)
    return (logits_grad,)

@partial(jax.custom_vjp, nondiff_argnums=(1,))
def straight_through_decode(logits, temperature=0.1):
    """
    Straight-through estimator for decoding.

    Forward: hard argmax decode (EXACTLY matches hard simulation)
    Backward: soft decode gradient (differentiable)
    """
    token = jnp.argmax(logits, axis=-1)
    return BINS[jnp.clip(token, 0, 1023)]

straight_through_decode.defvjp(_decode_fwd, _decode_bwd)


def _encode_embed_fwd(lataccel_hist, embed_weight, temperature):
    """Forward pass: exact hard token embedding lookup."""
    lataccel_hist_clipped = jnp.clip(lataccel_hist, -5, 5)
    hard_tokens = jnp.digitize(lataccel_hist_clipped, BINS, right=True)
    hard_token_emb = embed_weight[hard_tokens]
    # Store values needed for backward
    return hard_token_emb, (lataccel_hist_clipped, embed_weight)

def _encode_embed_bwd(temperature, res, g):
    """Backward pass: gradient through soft embedding.

    Note: When using nondiff_argnums, the non-diff args come first in bwd.
    """
    lataccel_hist_clipped, embed_weight = res

    def soft_embed_for_grad(lataccel_hist_clipped, embed_weight):
        # Soft encoding
        distances = (lataccel_hist_clipped[..., None] - BINS) ** 2
        logits = -distances / (2 * temperature ** 2)
        soft_probs = jax.nn.softmax(logits, axis=-1)
        # Soft embedding
        return jnp.einsum('...v,vd->...d', soft_probs, embed_weight)

    _, vjp_fn = jax.vjp(soft_embed_for_grad, lataccel_hist_clipped, embed_weight)
    lataccel_grad, embed_grad = vjp_fn(g)
    return (lataccel_grad, embed_grad)

@partial(jax.custom_vjp, nondiff_argnums=(2,))
def straight_through_encode_embed(lataccel_hist, embed_weight, temperature=0.1):
    """
    Straight-through estimator for encoding + embedding lookup.

    Forward: hard token lookup (EXACTLY matches hard simulation)
    Backward: gradient flows through soft encoding
    """
    lataccel_hist_clipped = jnp.clip(lataccel_hist, -5, 5)
    hard_tokens = jnp.digitize(lataccel_hist_clipped, BINS, right=True)
    return embed_weight[hard_tokens]

straight_through_encode_embed.defvjp(_encode_embed_fwd, _encode_embed_bwd)


def make_ste_step(model, temperature=0.1):
    """
    Create simulation step with straight-through estimator.

    Forward pass is EXACTLY the same as hard simulation.
    Backward pass uses soft approximations for gradient flow.
    """

    def simulation_step(carry, inputs):
        action_hist, lataccel_hist, exo_hist, current_lataccel = carry
        exo_row, action = inputs

        # Build states [batch, 20, 4]: (action, roll, v, a)
        # This is EXACTLY the same as hard simulation
        states = jnp.concatenate([
            action_hist[:, :, None],
            exo_hist,
        ], axis=-1)

        # Tokenize lataccel history - using STE for embedding
        # Forward: hard token lookup, Backward: soft gradient
        token_emb = straight_through_encode_embed(
            lataccel_hist, model.token_embed.weight, temperature
        )

        # State embedding (same as hard simulation)
        state_emb = states @ model.state_proj.weight.T + model.state_proj.bias

        # Concatenate and add position (same as hard simulation)
        B, T, _ = states.shape
        x = jnp.concatenate([state_emb, token_emb], axis=-1)
        x = x + model.pos_embed[:T]

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        # Transformer blocks (same as hard simulation)
        for block in model.blocks:
            x = block(x, mask)

        # Output (same as hard simulation)
        def ln(x, weight, bias, eps=1e-5):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + eps) * weight + bias

        x = ln(x, model.ln_f.weight, model.ln_f.bias)
        logits = x @ model.lm_head.weight.T

        # Straight-through decode: hard forward, soft backward
        next_lataccel = straight_through_decode(logits[:, -1, :], temperature)

        # Rate limit (same as hard simulation)
        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        # Update histories (same as hard simulation)
        action_hist = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
        exo_hist = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)
        lataccel_hist = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)

        new_carry = (action_hist, lataccel_hist, exo_hist, next_lataccel)
        return new_carry, next_lataccel

    return simulation_step


def run_simulation_ste(model, init_action_hist, init_lataccel_hist, init_exo_hist,
                       exo_data, actions, temperature=0.1):
    """
    Run simulation with straight-through estimator.

    Forward pass is EXACTLY identical to hard simulation.
    Backward pass uses soft approximations (differentiable).

    Args:
        model: TinyPhysicsModel (frozen weights)
        init_action_hist: [batch, 20]
        init_lataccel_hist: [batch, 20]
        init_exo_hist: [batch, 20, 3]
        exo_data: [n_steps, batch, 4] (roll, v, a, target)
        actions: [n_steps, batch]
        temperature: softmax temperature for gradient computation

    Returns:
        lataccels: [n_steps, batch] - predicted lateral accelerations
    """
    init_carry = (init_action_hist, init_lataccel_hist, init_exo_hist, init_lataccel_hist[:, -1])
    step_fn = make_ste_step(model, temperature=temperature)
    _, lataccels = lax.scan(step_fn, init_carry, (exo_data, actions))
    return lataccels


def make_ste_pid_step(model, temperature=0.1):
    """
    Create simulation step with STE and PID controller.

    Forward pass uses hard argmax (matches real inference).
    Backward pass uses soft approximations for gradient flow.
    PID params are passed in carry so they can receive gradients.
    """

    def simulation_step(carry, exo_row):
        (action_hist, lataccel_hist, exo_hist, current_lataccel,
         error_integral, prev_error, p, i, d) = carry

        # exo_row: [batch, 4] = roll, v, a, target
        target = exo_row[:, 3]

        # PID controller - compute action FIRST (like tinyphysics.py control_step)
        error = target - current_lataccel
        error_integral_new = error_integral + error
        error_derivative = error - prev_error
        action = p * error + i * error_integral_new + d * error_derivative
        action = jnp.clip(action, -2.0, 2.0)

        # Update action history BEFORE model forward (tinyphysics.py does control_step then sim_step)
        # The current step's action must be in the history when the model runs
        action_hist_updated = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
        exo_hist_updated = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)

        # Build states [batch, 20, 4]: (action, roll, v, a) - with CURRENT action included
        states = jnp.concatenate([
            action_hist_updated[:, :, None],
            exo_hist_updated,
        ], axis=-1)

        # Tokenize lataccel history - using STE for embedding
        token_emb = straight_through_encode_embed(
            lataccel_hist, model.token_embed.weight, temperature
        )

        # State embedding
        state_emb = states @ model.state_proj.weight.T + model.state_proj.bias

        # Concatenate and add position
        B, T, _ = states.shape
        x = jnp.concatenate([state_emb, token_emb], axis=-1)
        x = x + model.pos_embed[:T]

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        # Transformer blocks
        for block in model.blocks:
            x = block(x, mask)

        # Output
        def ln(x, weight, bias, eps=1e-5):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + eps) * weight + bias

        x = ln(x, model.ln_f.weight, model.ln_f.bias)
        logits = x @ model.lm_head.weight.T

        # Straight-through decode: hard forward, soft backward
        next_lataccel = straight_through_decode(logits[:, -1, :], temperature)

        # Rate limit
        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        # Update lataccel history (action/exo already updated above before model forward)
        lataccel_hist_updated = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)

        new_carry = (action_hist_updated, lataccel_hist_updated, exo_hist_updated, next_lataccel,
                     error_integral_new, error, p, i, d)

        output = jnp.stack([next_lataccel, action, target], axis=-1)  # [batch, 3]
        return new_carry, output

    return simulation_step


def run_simulation_ste_pid(model, init_action_hist, init_lataccel_hist, init_exo_hist,
                           exo_data, p, i, d, init_error_integral=None, init_prev_error=None,
                           temperature=0.1):
    """
    Run simulation with STE and PID controller.

    Differentiable w.r.t. PID gains p, i, d.

    Args:
        model: TinyPhysicsModel
        init_action_hist: [batch, 20]
        init_lataccel_hist: [batch, 20]
        init_exo_hist: [batch, 20, 3]
        exo_data: [n_steps, batch, 4] (roll, v, a, target)
        p, i, d: PID gains (scalars or arrays broadcastable to batch)
        init_error_integral: [batch] initial accumulated error (from warmup)
        init_prev_error: [batch] previous error (from warmup)
        temperature: softmax temperature for STE

    Returns:
        outputs: [n_steps, batch, 3] (lataccel, action, target)
    """
    batch_size = init_action_hist.shape[0]

    if init_error_integral is None:
        init_error_integral = jnp.zeros(batch_size)
    if init_prev_error is None:
        init_prev_error = jnp.zeros(batch_size)

    init_carry = (
        init_action_hist,
        init_lataccel_hist,
        init_exo_hist,
        init_lataccel_hist[:, -1],
        init_error_integral,
        init_prev_error,
        p, i, d
    )

    step_fn = make_ste_pid_step(model, temperature=temperature)
    _, outputs = jax.lax.scan(step_fn, init_carry, exo_data)
    return outputs  # [n_steps, batch, 3]


# ============================================================================
# FULLY DIFFERENTIABLE SOFT PID SIMULATION
# Uses soft encode/decode throughout - gradients match finite differences
# ============================================================================

def make_soft_pid_step(model, temperature=0.5):
    """
    Create fully differentiable simulation step with PID controller.

    Uses soft encoding/decoding throughout so gradients are correct.
    """

    def simulation_step(carry, exo_row):
        (action_hist, lataccel_hist, exo_hist, current_lataccel,
         error_integral, prev_error, p, i, d) = carry

        # exo_row: [batch, 4] = roll, v, a, target
        target = exo_row[:, 3]

        # PID controller
        error = target - current_lataccel
        error_integral_new = error_integral + error
        error_derivative = error - prev_error
        action = p * error + i * error_integral_new + d * error_derivative
        action = jnp.clip(action, -2.0, 2.0)

        # Update histories BEFORE model forward
        action_hist_updated = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
        exo_hist_updated = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)

        # Build states [batch, 20, 4]: (action, roll, v, a)
        states = jnp.concatenate([
            action_hist_updated[:, :, None],
            exo_hist_updated,
        ], axis=-1)

        # Soft forward pass (fully differentiable)
        logits = model_forward_soft(model, states, lataccel_hist, temperature)

        # Soft decode (fully differentiable)
        next_lataccel = soft_decode(logits[:, -1, :], temperature)

        # Rate limit
        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        # Update lataccel history
        lataccel_hist_updated = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)

        new_carry = (action_hist_updated, lataccel_hist_updated, exo_hist_updated, next_lataccel,
                     error_integral_new, error, p, i, d)

        output = jnp.stack([next_lataccel, action, target], axis=-1)
        return new_carry, output

    return simulation_step


def run_simulation_soft_pid(model, init_action_hist, init_lataccel_hist, init_exo_hist,
                            exo_data, p, i, d, init_error_integral=None, init_prev_error=None,
                            temperature=0.5):
    """
    Run fully differentiable simulation with PID controller.

    Gradients are correct (match finite differences).
    Use for optimization, then evaluate with run_simulation_ste_pid.

    Args:
        model: TinyPhysicsModel
        init_action_hist: [batch, 20]
        init_lataccel_hist: [batch, 20]
        init_exo_hist: [batch, 20, 3]
        exo_data: [n_steps, batch, 4] (roll, v, a, target)
        p, i, d: PID gains
        init_error_integral: [batch] initial accumulated error (from warmup)
        init_prev_error: [batch] previous error (from warmup)
        temperature: softmax temperature (higher = smoother)

    Returns:
        outputs: [n_steps, batch, 3] (lataccel, action, target)
    """
    batch_size = init_action_hist.shape[0]

    if init_error_integral is None:
        init_error_integral = jnp.zeros(batch_size)
    if init_prev_error is None:
        init_prev_error = jnp.zeros(batch_size)

    init_carry = (
        init_action_hist,
        init_lataccel_hist,
        init_exo_hist,
        init_lataccel_hist[:, -1],
        init_error_integral,
        init_prev_error,
        p, i, d
    )

    step_fn = make_soft_pid_step(model, temperature=temperature)
    _, outputs = jax.lax.scan(step_fn, init_carry, exo_data)
    return outputs


# ============================================================================
# CROSS-ENTROPY PID SIMULATION
# Uses cross-entropy loss on logits - gradients match finite differences!
# ============================================================================

def make_ce_pid_step(model, temperature=0.5):
    """
    Create simulation step with cross-entropy loss on logits.

    Returns logits for CE loss computation, uses soft decode for recurrence.
    """

    def simulation_step(carry, exo_row):
        (action_hist, lataccel_hist, exo_hist, current_lataccel,
         error_integral, prev_error, p, i, d) = carry

        target = exo_row[:, 3]

        # PID controller
        error = target - current_lataccel
        error_integral_new = error_integral + error
        error_derivative = error - prev_error
        action = p * error + i * error_integral_new + d * error_derivative
        action = jnp.clip(action, -2.0, 2.0)

        # Update histories BEFORE model forward
        action_hist_updated = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
        exo_hist_updated = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)

        # Build states
        states = jnp.concatenate([
            action_hist_updated[:, :, None],
            exo_hist_updated,
        ], axis=-1)

        # Soft forward pass
        logits = model_forward_soft(model, states, lataccel_hist, temperature)

        # Soft decode for recurrence
        next_lataccel = soft_decode(logits[:, -1, :], temperature)
        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        # Update lataccel history
        lataccel_hist_updated = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)

        new_carry = (action_hist_updated, lataccel_hist_updated, exo_hist_updated, next_lataccel,
                     error_integral_new, error, p, i, d)

        # Output: logits (for CE loss), soft lataccel, action, target
        output = (logits[:, -1, :], next_lataccel, action, target)
        return new_carry, output

    return simulation_step


def run_simulation_ce_pid(model, init_action_hist, init_lataccel_hist, init_exo_hist,
                          exo_data, p, i, d, init_error_integral=None, init_prev_error=None,
                          temperature=0.5):
    """
    Run simulation returning logits for cross-entropy loss.

    Returns:
        logits: [n_steps, batch, 1024] - raw logits for CE loss
        lataccels: [n_steps, batch] - soft decoded values
        actions: [n_steps, batch] - PID actions
        targets: [n_steps, batch] - target lataccels
    """
    batch_size = init_action_hist.shape[0]

    if init_error_integral is None:
        init_error_integral = jnp.zeros(batch_size)
    if init_prev_error is None:
        init_prev_error = jnp.zeros(batch_size)

    init_carry = (
        init_action_hist,
        init_lataccel_hist,
        init_exo_hist,
        init_lataccel_hist[:, -1],
        init_error_integral,
        init_prev_error,
        p, i, d
    )

    step_fn = make_ce_pid_step(model, temperature=temperature)
    _, outputs = jax.lax.scan(step_fn, init_carry, exo_data)

    logits, lataccels, actions, targets = outputs
    return logits, lataccels, actions, targets


# ============================================================================
# SPARSEMAX PID SIMULATION
# Uses sparsemax for both encoding and decoding - sparse but differentiable
# Output often matches hard argmax exactly while still having gradients
# ============================================================================

def model_forward_sparsemax(model, states, lataccel_hist):
    """Forward pass with sparsemax token embeddings (differentiable, sparse)."""
    B, T, _ = states.shape

    # Sparsemax encode lataccel history
    # For encoding, we use hard tokens since sparsemax encoding is expensive
    # and the main benefit is in the decode step
    lataccel_hist_clipped = jnp.clip(lataccel_hist, -5, 5)
    hard_tokens = jnp.digitize(lataccel_hist_clipped, BINS, right=True)
    token_emb = model.token_embed.weight[hard_tokens]

    # State embedding
    state_emb = states @ model.state_proj.weight.T + model.state_proj.bias

    # Concatenate and add position
    x = jnp.concatenate([state_emb, token_emb], axis=-1)
    x = x + model.pos_embed[:T]

    # Causal mask
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))

    # Transformer blocks
    for block in model.blocks:
        x = block(x, mask)

    # Output
    def ln(x, weight, bias, eps=1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + eps) * weight + bias

    x = ln(x, model.ln_f.weight, model.ln_f.bias)
    logits = x @ model.lm_head.weight.T

    return logits


def make_sparsemax_pid_step(model):
    """
    Create simulation step with sparsemax decoding.

    Uses hard encoding (fast) but sparsemax decoding (differentiable, often exact).
    """

    def simulation_step(carry, exo_row):
        (action_hist, lataccel_hist, exo_hist, current_lataccel,
         error_integral, prev_error, p, i, d) = carry

        target = exo_row[:, 3]

        # PID controller
        error = target - current_lataccel
        error_integral_new = error_integral + error
        error_derivative = error - prev_error
        action = p * error + i * error_integral_new + d * error_derivative
        action = jnp.clip(action, -2.0, 2.0)

        # Update histories BEFORE model forward
        action_hist_updated = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
        exo_hist_updated = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)

        # Build states
        states = jnp.concatenate([
            action_hist_updated[:, :, None],
            exo_hist_updated,
        ], axis=-1)

        # Forward pass (uses hard encoding for efficiency)
        logits = model_forward_sparsemax(model, states, lataccel_hist)

        # Sparsemax decode - sparse but differentiable
        next_lataccel = sparsemax_decode(logits[:, -1, :])

        # Rate limit
        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

        # Update lataccel history
        lataccel_hist_updated = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)

        new_carry = (action_hist_updated, lataccel_hist_updated, exo_hist_updated, next_lataccel,
                     error_integral_new, error, p, i, d)

        output = jnp.stack([next_lataccel, action, target], axis=-1)
        return new_carry, output

    return simulation_step


def run_simulation_sparsemax_pid(model, init_action_hist, init_lataccel_hist, init_exo_hist,
                                  exo_data, p, i, d, init_error_integral=None, init_prev_error=None):
    """
    Run simulation with sparsemax decoding.

    Sparsemax often produces exact one-hot outputs (matching hard argmax)
    while still being differentiable at the boundaries.

    Args:
        model: TinyPhysicsModel
        init_action_hist: [batch, 20]
        init_lataccel_hist: [batch, 20]
        init_exo_hist: [batch, 20, 3]
        exo_data: [n_steps, batch, 4] (roll, v, a, target)
        p, i, d: PID gains
        init_error_integral: [batch] initial accumulated error (from warmup)
        init_prev_error: [batch] previous error (from warmup)

    Returns:
        outputs: [n_steps, batch, 3] (lataccel, action, target)
    """
    batch_size = init_action_hist.shape[0]

    if init_error_integral is None:
        init_error_integral = jnp.zeros(batch_size)
    if init_prev_error is None:
        init_prev_error = jnp.zeros(batch_size)

    init_carry = (
        init_action_hist,
        init_lataccel_hist,
        init_exo_hist,
        init_lataccel_hist[:, -1],
        init_error_integral,
        init_prev_error,
        p, i, d
    )

    step_fn = make_sparsemax_pid_step(model)
    _, outputs = jax.lax.scan(step_fn, init_carry, exo_data)
    return outputs
