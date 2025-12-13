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
class TinyPhysicsModel(eqx.Module):
    """TinyPhysics GPT-style model."""

    state_proj: nn.Linear
    token_embed: nn.Embedding
    pos_embed: jax.Array
    blocks: list
    ln_f: nn.LayerNorm
    lm_head: nn.Linear
    context_len: int = eqx.field(static=True)

    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 128,
        d_ff: int = 512,
        context_len: int = 20,
        vocab_size: int = 1024,
        state_dim: int = 4,
        *,
        key
    ):
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


def load_weights_from_onnx(model: TinyPhysicsModel, onnx_path: str) -> TinyPhysicsModel:
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
    model = eqx.tree_at(
        lambda m: m.lm_head.weight,
        model,
        jnp.array(weights['onnx::MatMul_570'].T)
    )

    return model


def create_model(onnx_path: str) -> TinyPhysicsModel:
    """Create model and load weights from ONNX."""
    key = jax.random.PRNGKey(0)
    model = TinyPhysicsModel(key=key)
    model = load_weights_from_onnx(model, onnx_path)
    return model


# Tokenization
BINS = jnp.linspace(-5, 5, 1024)
MAX_ACC_DELTA = 0.5
CONTEXT_LENGTH = 20

def encode(value):
    """Encode lataccel to token."""
    value = jnp.clip(value, -5, 5)
    return jnp.digitize(value, BINS, right=True)

def decode(token):
    """Decode token to lataccel."""
    return BINS[jnp.clip(token, 0, 1023)]


def make_simulation_step(model):
    """Create a simulation step function for use with lax.scan."""

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
