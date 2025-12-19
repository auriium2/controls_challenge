"""Test that JAX simulation matches ONNX simulation."""
import numpy as np
import pandas as pd
import jax.numpy as jnp
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTEXT_LENGTH, CONTROL_START_IDX, ACC_G
from tinyphysics_eqx import create_model, run_simulation_ste_pid
from controllers.pid import Controller

# Load one file
data_file = "data/00000.csv"
df = pd.read_csv(data_file)

# Run ONNX simulation
print("Running ONNX simulation...")
onnx_model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
controller = Controller()
sim = TinyPhysicsSimulator(onnx_model, data_file, controller=controller, debug=False)

# Set deterministic seed
np.random.seed(42)
cost = sim.rollout()

onnx_lataccels = np.array(sim.current_lataccel_history)
onnx_actions = np.array(sim.action_history)
print(f"ONNX cost: {cost}")
print(f"ONNX lataccels at control start [100:105]: {onnx_lataccels[100:105]}")
print(f"ONNX actions at control start [100:105]: {onnx_actions[100:105]}")

# Run JAX simulation
print("\nRunning JAX simulation...")
jax_model = create_model("models/tinyphysics.onnx")

# Prepare data like optimize_pid.py does
roll_lataccel = np.sin(df['roll'].values) * ACC_G
v_ego = df['vEgo'].values
a_ego = df['aEgo'].values
target = df['targetLateralAcceleration'].values
steer = -df['steerCommand'].values

exo = np.stack([roll_lataccel, v_ego, a_ego, target], axis=-1)

# Compute accumulated PID state from warmup
# During warmup, current_lataccel[t] = target[t-1] (1-step lag)
error_integral = 0.0
prev_error = 0.0
for t in range(CONTEXT_LENGTH, CONTROL_START_IDX):
    current = target[t-1]
    error = target[t] - current
    error_integral += error
    prev_error = error

print(f"Computed error_integral from warmup: {error_integral}")
print(f"Computed prev_error from warmup: {prev_error}")

# At step 100, we need last 20 steps of warmup state
warmup_end = CONTROL_START_IDX
init_action_hist = steer[warmup_end-CONTEXT_LENGTH:warmup_end]
init_lataccel_hist = target[warmup_end-CONTEXT_LENGTH:warmup_end]
init_exo_hist = exo[warmup_end-CONTEXT_LENGTH:warmup_end, :3]

# Control period exo data
control_horizon = 400  # steps 100-500
control_exo = exo[CONTROL_START_IDX:CONTROL_START_IDX+control_horizon]

# Make batch dimension
init_action_hist = jnp.array(init_action_hist)[None, :]
init_lataccel_hist = jnp.array(init_lataccel_hist)[None, :]
init_exo_hist = jnp.array(init_exo_hist)[None, :, :]
control_exo = jnp.array(control_exo)[None, :, :].transpose(1, 0, 2)
init_error_integral = jnp.array([error_integral])
init_prev_error = jnp.array([prev_error])

# PID params (same as actual controller)
p, i, d = 0.195, 0.100, -0.053

outputs = run_simulation_ste_pid(
    jax_model,
    init_action_hist,
    init_lataccel_hist,
    init_exo_hist,
    control_exo,
    p, i, d,
    init_error_integral=init_error_integral,
    init_prev_error=init_prev_error,
    temperature=0.1
)

jax_lataccels = np.array(outputs[:, 0, 0])
jax_actions = np.array(outputs[:, 0, 1])

print(f"\nJAX lataccels [0:5]: {jax_lataccels[:5]}")
print(f"JAX actions [0:5]: {jax_actions[:5]}")

# Compare
print(f"\n=== Comparison ===")
print(f"ONNX lataccels[100:105]: {onnx_lataccels[100:105]}")
print(f"JAX  lataccels[0:5]:     {jax_lataccels[:5]}")

print(f"\nONNX actions[100:105]: {onnx_actions[100:105]}")
print(f"JAX  actions[0:5]:     {jax_actions[:5]}")

# Check if they match
lataccel_match = np.allclose(onnx_lataccels[100:105], jax_lataccels[:5], atol=1e-4)
action_match = np.allclose(onnx_actions[100:105], jax_actions[:5], atol=1e-4)
print(f"\nLataccels match: {lataccel_match}")
print(f"Actions match: {action_match}")
