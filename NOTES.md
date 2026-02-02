# Controls Challenge - Working Notes

## Project Overview

Comma Controls Challenge v2: write a controller that drives a simulated car to follow a desired lateral acceleration trajectory. The simulator is a GPT-style autoregressive model (TinyPhysics) that predicts lateral acceleration given steering commands.

## Architecture

### TinyPhysics Model
- GPT-style transformer: 4 layers, 4 heads, d_model=128, d_ff=512, context_len=20
- Input: `states [batch, 20, 4]` = (action, roll_lataccel, v_ego, a_ego) + `tokens [batch, 20]` = tokenized lataccel history
- Output: `logits [batch, 20, 1024]` over 1024 bins in [-5, 5]
- Decoding: original uses sampling with temperature=0.8, seeded by md5(data_path)
- Rate limit: `next_lataccel = clip(pred, current - 0.5, current + 0.5)`
- Action clipped to [-2, 2] (STEER_RANGE)

### Tokenization
- 1024 bins linearly spaced in [-5, 5] (`common.py: BINS`)
- Encode: `jnp.digitize(clip(value, -5, 5), BINS, right=True)` 
- Decode: `BINS[token]`
- Piecewise constant - zero gradient, and that's **correct** (not a limitation)

### Simulator Loop (tinyphysics.py)
```
for step in range(CONTEXT_LENGTH, len(data)):
    1. state, target, futureplan = get_state_target_futureplan(step)
    2. state_history.append(state)
    3. target_lataccel_history.append(target)
    4. control_step(): action = controller.update(...); clip to [-2,2]; action_history.append(action)
    5. sim_step(): run model on last 20 of (action_history, state_history, lataccel_history)
       - Before step 100: use ground truth target as current_lataccel
       - From step 100+: use model prediction
```

### Cost Function
- `lataccel_cost = mean((target - pred)^2) * 100` (steps 100-500)
- `jerk_cost = mean((diff(pred) / 0.1)^2) * 100` (steps 100-500)
- `total_cost = lataccel_cost * 50 + jerk_cost`
- Tracking weighted 50x more than jerk

### Controller Interface
```python
class BaseController:
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # state: State(roll_lataccel, v_ego, a_ego)
        # future_plan: FuturePlan(lataccel[50], roll_lataccel[50], v_ego[50], a_ego[50])
        return action  # scalar
```

### Key Constants
- CONTEXT_LENGTH = 20 (history window)
- CONTROL_START_IDX = 100 (controller takes effect)
- COST_END_IDX = 500 (cost computed over 100-500)
- FUTURE_PLAN_STEPS = 50 (5 seconds at 10 FPS)
- MAX_ACC_DELTA = 0.5 (rate limit per step)
- DEL_T = 0.1 (timestep)
- TEMPERATURE = 0.8

## Equinox Clone (tinyphysics_eqx.py)

JAX/Equinox reimplementation of the TinyPhysics model. Verified to produce **identical results** to the ONNX model:
- Forward pass logits match to <1e-3
- Full rollout trajectories match exactly (argmax decode)
- Integration tested: dropping EQX into TinyPhysicsSimulator (via wrapper) produces identical costs, trajectories, and actions as ONNX (test_physics_parity.py, 17/17 tests pass)

### Existing simulation functions in tinyphysics_eqx.py
- `run_simulation()` - open-loop with given actions (argmax decode)
- `run_simulation_pid()` - PID controller in the loop (argmax)
- `run_simulation_ste()` - straight-through estimator (hard fwd, soft bwd)
- `run_simulation_soft_pid()` - fully soft encode/decode (for gradient flow)
- `run_simulation_sparsemax_pid()` - sparsemax decode
- Various other variants for different differentiability strategies

## Key Insight: Differentiability for iLQR

The transformer can be viewed as: `f(states_tensor, token_embeddings) -> logits`

**Encoder (lataccel -> token -> embedding):**
- `jnp.digitize` is piecewise constant. Its true local Jacobian is zero.
- This is **correct**, not an approximation. Small perturbations within a bin don't change the token.
- No STE needed. Integer indexing into embedding table is naturally non-differentiable in JAX.
- Gradients through the token/embedding path are correctly zero.

**Decoder (logits -> lataccel):**
- Replace argmax with expected value: `E[lataccel] = softmax(logits/T) @ BINS`
- Naturally differentiable w.r.t. logits (and therefore w.r.t. action input)
- Good surrogate for argmax since distributions are typically peaked

**Clips (action [-2,2], rate limit [current-0.5, current+0.5]):**
- Use hard `jnp.clip`. Grad=1 when unsaturated, 0 at boundary.
- Both are correct information for iLQR. No soft clip needed.

**Gradient flow path:** `action -> state_proj -> transformer -> logits -> EV decode -> next_lataccel`
The lataccel_hist path through tokens contributes zero gradient (correct). The lataccel_hist still matters for jerk cost and rate limiter (through `current_lataccel`).

## iLQR Controller (controllers/ilqr.py)

### Status: Partially working, performance issues

The iLQR controller runs and produces different actions from PID (verified output shows `u_opt[0]=-0.0816` vs `pid=0.0013`), but hangs after the first few steps. This is likely because JIT compilation of the full `ilqr_solve` (which includes `jax.jacobian` of the dynamics inside `lax.scan`) is extremely slow on first trace.

### State Space (R^41)
- `x[0:20]` = action_hist (last 20 actions)
- `x[20:40]` = lataccel_hist (last 20 predicted lataccels)
- `x[40]` = current_lataccel

### Dynamics
```python
def dynamics_flat(x, u, exo_hist, exo_next):
    # Clip action, shift histories, build states tensor
    # Call model(states, tokens) using existing EQX model.__call__
    # EV decode: softmax(logits/T) @ BINS
    # Rate limit clip
    # Return flat next state [41]
```

### Jacobian Strategy
- `jax.jacobian(dynamics_flat, argnums=(0, 1))` gives A [41,41] and B [41]
- Single call handles everything: shifts, encode, transformer, EV decode, clips
- Only 1 Jacobian per timestep (not 41 separate grads)

### Algorithm
Standard iLQR with:
- Forward rollout via lax.scan
- Backward pass with Jacobians at each step, scalar Q_uu (no matrix inversion needed)
- Line search over alphas [1.0, 0.5, 0.25, 0.1]
- Regularization mu with adaptive updates
- Warm start: shift previous solution

### Known Issues
1. **Metal backend incompatible**: `mhlo.dot_general` legalization error when JIT-compiling Jacobians. Must use CPU (`JAX_PLATFORMS=cpu`).
2. **JIT compilation extremely slow**: The first call triggers tracing of nested `lax.scan` with `jax.jacobian` inside. The 5 iLQR iterations x 20 horizon steps x Jacobian computation creates a massive computation graph. First 5 steps work (likely cached from trace), then hangs on recompilation or execution.
3. **Potential fixes to explore**:
   - Replace inner lax.scan loops with Python for-loops (eager execution, no tracing overhead)
   - Pre-compile Jacobian function separately, cache it
   - Reduce horizon (try H=5 or H=10)
   - Reduce iterations (try n_iters=2)
   - Use `jax.jacrev` on just the scalar `next_lataccel` output and construct A,B analytically from shift structure + scalar gradient (1 vjp per step instead of full 41x42 Jacobian)

### Controller Integration
- History bootstrapping via stack inspection on first call (gets histories from TinyPhysicsSimulator)
- Incremental history updates on subsequent calls
- PID fallback for NaN or exceptions
- Warm start: shift previous solution for next call

## Test Suite

### tests/test_weights.py (existing, comprehensive)
- Forward pass parity: ONNX vs EQX logits and tokens match
- Full rollout parity: open-loop and PID, real and random actions
- Edge cases: extreme actions, exo values, tokens
- Tokenization: encode/decode parity and roundtrip
- Batched operations: batched vs sequential consistency
- PID simulation: step-by-step ONNX vs EQX PID rollout
- STE simulation: STE forward matches hard simulation exactly
- Against TinyPhysicsSimulator: full eval pipeline comparison

### tests/test_physics_parity.py (new, integration)
- Wraps EQX model in `EQXModelWrapper` that presents same interface as `TinyPhysicsModel`
- Drops into `TinyPhysicsSimulator` directly - same seed, same sampling, same warmup
- Tests: PID rollout trajectories identical, costs identical, actions identical
- Also tests zero controller parity
- **All 17 tests pass**

## File Layout
```
controllers/
  __init__.py       - BaseController
  pid.py            - PID controller (p=0.195, i=0.100, d=-0.053)
  zero.py           - Zero controller (always returns 0)
  ilqr.py           - iLQR controller (WIP)
tinyphysics.py      - Original simulator + eval pipeline (ONNX)
tinyphysics_eqx.py  - Equinox model clone + various simulation functions
common.py           - BINS, TEMPERATURE, MAX_ACC_DELTA, encode, decode
tests/
  test_weights.py          - Comprehensive ONNX vs EQX parity tests
  test_physics_parity.py   - Integration test: full eval pipeline parity
```

## Next Steps

1. **Fix iLQR performance**: The JIT compilation of the full solver is too slow. Most promising approach: replace `lax.scan` loops with Python for-loops so the Jacobian computation runs eagerly. This avoids the massive trace/compile overhead while still using JAX autodiff for each individual Jacobian call.

2. **Validate iLQR correctness**: Once it runs at reasonable speed, verify that:
   - Jacobians are correct (finite-difference check)
   - Cost decreases across iLQR iterations
   - Actions differ meaningfully from PID
   - Total cost improves over PID baseline

3. **Tune iLQR parameters**:
   - Horizon (start with 10, try 20)
   - Number of iterations (start with 3)
   - Cost weights (w_tracking=50, w_jerk=1, w_action=0.01)
   - Regularization schedule

4. **Benchmark**: Run `eval.py` with iLQR vs PID on 100+ segments to measure improvement.
