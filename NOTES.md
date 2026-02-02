# Controls Challenge - Important Notes

## Current Status (2024-12-13)

We have a working offline training pipeline for the bicycle model using pre-collected trajectory data.

**Latest trained model results (extended model, clean data):**
- Loss: 0.053 (down from 58 at start)
- Parameters:
  - steer_ratio=0.019 (very small - steering has minimal effect)
  - wheelbase=3.02m
  - roll_coeff=0.88
  - time_constant=0.33s
  - v_steer=-0.025
  - accel=0.019
  - bias=-0.18

**Key finding:** The transformer predicts lataccel primarily from roll, not steering. The steer_ratio consistently converges to ~0.02 regardless of training setup.

## The Goal

Train a differentiable bicycle model that matches the transformer's behavior. This lets us:
1. Use the bicycle model as a fast differentiable simulator
2. Train controllers that work on the real transformer
3. Potentially use STE to backprop through the full system

## Data Format

CSV columns: `t, vEgo, aEgo, roll, targetLateralAcceleration, steerCommand`

**CRITICAL**: `steerCommand` is NaN after index 100. The first 100 steps are warmup with ground truth steering. After that, YOUR CONTROLLER provides the steering.

- `CONTEXT_LENGTH = 20` - model uses 20 steps of history
- `CONTROL_START_IDX = 100` - controller takes over at step 100
- `COST_END_IDX = 500` - evaluation ends at step 500
- **20,000 files available** in data/

## Data Preprocessing (from tinyphysics.py)

```python
'roll_lataccel': np.sin(df['roll'].values) * ACC_G,  # NOT raw roll!
'steer_command': -df['steerCommand'].values,  # NEGATED!
```

- Roll is converted: `sin(roll) * 9.81`
- Steer is negated (left-positive to right-positive convention)

## Saturation Issues - CRITICAL

The system has hard limits that cause saturation:
- `LATACCEL_RANGE = [-5, 5]` - lataccel clipped for tokenization
- `STEER_RANGE = [-2, 2]` - actions clipped
- `MAX_ACC_DELTA = 0.5` - rate limiting on lataccel changes

**Impact of saturation on training:**
- Saturated data: loss = 0.118, time_constant = 0.50s
- Clean data (no saturation): loss = 0.053, time_constant = 0.33s
- **55% loss improvement** just by removing saturated trajectories!

**Why saturation hurts:**
1. Flat gradients in clipped regions
2. Model learns incorrect dynamics (clipped values persist longer, making it think response is slower)
3. Time constant was artificially inflated by 50% due to saturation

**Solution:** Filter out any trajectory where lataccel hits exactly ±5 or action hits exactly ±2.
- Only 2.3% of trajectories saturate with varied PID gains
- Keep 97.7% of data, get much better model

## Other Potential Data Quality Issues to Watch

1. **Rate limiting (MAX_ACC_DELTA = 0.5)**: Could also cause similar artifacts if model predictions frequently hit the rate limit
2. **Tokenization quantization**: 1024 bins from -5 to 5 = ~0.01 resolution. Small values may be quantized away
3. **PID gain distribution**: If certain PID gains dominate, model may overfit to those action patterns
4. **Velocity distribution**: If training data has narrow velocity range, model may not generalize to other speeds
5. **Roll distribution**: If roll values are correlated with steering in training data, model may learn spurious relationships

## The Transformer Model

- Trained to mimic a bicycle model simulator
- Inputs: (action, roll_lataccel, v_ego, a_ego) states + tokenized lataccel history
- Output: logits over 1024 bins for next lataccel
- Uses temperature=0.8 sampling in original, we use argmax for deterministic
- Tokenization: 1024 bins from -5 to 5

## Bicycle Model Physics

```
lateral_accel = (v^2 / wheelbase) * steering_angle + roll_contribution
```

With first-order lag dynamics. Learnable parameters:
- `steer_ratio`: converts steer command to wheel angle (rad per unit)
- `wheelbase`: effective wheelbase (meters)
- `roll_coeff`: roll sensitivity multiplier
- `time_constant`: first-order response lag (seconds)

Extended model adds:
- `v_steer_coeff`: velocity modifies steering effectiveness
- `accel_coeff`: longitudinal acceleration contribution
- `bias`: constant offset

Small angle approximation used: `curvature = delta / wheelbase` instead of `tan(delta) / wheelbase`

## Training Approach

**Offline training (fast):**
1. Pre-collect trajectories using `modal_collect_training_data.py` on H100 (253k samples/sec)
2. Filter out saturated trajectories
3. Train bicycle model on static data using `train_bicycle_offline.py`
4. No transformer inference during training - just bicycle gradients

**Data collection:**
- Run warmup (steps 0-99) with CSV actions
- Run PID with varied gains (p: 0.1-0.3, i: 0.05-0.15, d: -0.1 to -0.02)
- Store: transformer_lataccels, pid_actions, init_lataccel, exo_data

**Why offline is better:**
- Transformer inference is expensive
- Gradient only flows through bicycle (4-7 params)
- Can use huge batch sizes (2048+)
- 50k trajectories trains in ~2 minutes locally

## STE (Straight-Through Estimator)

Implemented in tinyphysics_eqx.py using `jax.custom_vjp`:
- Forward: exact hard token decode (matches ONNX exactly)
- Backward: soft gradient for differentiability

All 112 parity tests pass. STE is available for future use (e.g., training controllers through the full system).

## Key Files

- `tinyphysics.py` - Original ONNX-based simulator (reference implementation)
- `tinyphysics_eqx.py` - JAX/Equinox clone with STE support
- `bicycle_model.py` - Physics-based bicycle dynamics (basic + extended)
- `train_bicycle_offline.py` - Fast offline training on pre-collected data
- `modal_collect_training_data.py` - Collect trajectory data on H100
- `eval.py` - Evaluation script for controllers

## Rollout Behavior (from tinyphysics.py)

```python
# During warmup (step < 100):
self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]  # Use ground truth

# After warmup (step >= 100):
self.current_lataccel = pred  # Use model prediction
```

During warmup, lataccel history is filled with ground truth from CSV, not model predictions.

## Training Infrastructure

- **Local:** Apple M4 Max with Metal (JAX experimental support)
- **Remote:** Modal H100 for data collection (253k samples/sec)
- **Data:** 20,000 CSV files, 50k pre-collected trajectories

## Common Mistakes Made (Don't Repeat!)

1. Using raw `roll` instead of `sin(roll) * ACC_G`
2. Forgetting to negate steerCommand
3. Using steerCommand after index 100 (it's NaN!)
4. Using MLP instead of physics-based bicycle model
5. Training on single-step transitions instead of rollouts
6. Using `tan()` which explodes - use small angle approximation
7. Running PID separately through bicycle (actions depend on lataccel feedback!)
8. **Training on saturated data** - causes 55% higher loss and wrong time constant!
9. Computing gradients through transformer (expensive and unnecessary)

## Next Steps

1. Investigate why steer_ratio → 0 (is transformer ignoring steering?)
2. Use trained bicycle model for controller optimization
3. Consider checking rate-limit saturation (MAX_ACC_DELTA = 0.5)
4. Validate bicycle model on held-out trajectories
