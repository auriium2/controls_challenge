"""
Modal script for collecting bicycle training data with diverse action generators.

Action generators:
1. Varied PID gains
2. Lazy PID (weak gains)  
3. Smoothed random walk
4. Sinusoidal sweeps
5. Scaled target tracking

Usage:
    modal run modal_collect_diverse.py --n-trajectories 100000
"""

import modal

app = modal.App("bicycle-training-data-diverse")

volume = modal.Volume.from_name("koopman-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]==0.6.0",
        "equinox>=0.11.0",
        "onnx>=1.15.0",
        "numpy>=1.24.0",
    )
    .add_local_file("models/tinyphysics.onnx", remote_path="/root/models/tinyphysics.onnx")
    .add_local_file("tinyphysics_eqx.py", remote_path="/root/tinyphysics_eqx.py")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/data": volume},
)
def collect_training_data(n_trajectories: int, horizon: int, batch_size: int, seed: int) -> bytes:
    """Collect training data with diverse action generators."""
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from jax import lax
    import time
    import io

    from tinyphysics_eqx import (
        create_model, run_simulation, run_simulation_pid,
        CONTEXT_LENGTH, MAX_ACC_DELTA, encode, decode
    )

    CONTROL_START_IDX = 100

    print(f"JAX devices: {jax.devices()}")
    print(f"Collecting {n_trajectories} trajectories, horizon={horizon}, batch_size={batch_size}")

    # Load driving data
    driving_data = np.load("/data/driving_data.npy")
    n_files, max_steps, _ = driving_data.shape
    print(f"Loaded driving data: {driving_data.shape}")

    available_horizon = max_steps - CONTROL_START_IDX
    if horizon > available_horizon:
        horizon = available_horizon
        print(f"Clamped horizon to {horizon}")

    model = create_model("/root/models/tinyphysics.onnx")

    # =========================================================================
    # Batched simulation step (shared by all controllers)
    # =========================================================================
    
    def make_sim_step(model):
        """Simulation step that takes action as input."""
        def step(carry, inputs):
            action_hist, lataccel_hist, exo_hist, current_lataccel = carry
            exo_row, action = inputs  # exo_row: [batch, 4], action: [batch]
            
            states = jnp.concatenate([
                action_hist[:, :, None],
                exo_hist,
            ], axis=-1)
            
            tokens = encode(lataccel_hist)
            logits = model(states, tokens)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1)
            next_lataccel = decode(next_token)
            next_lataccel = jnp.clip(next_lataccel, 
                                      current_lataccel - MAX_ACC_DELTA,
                                      current_lataccel + MAX_ACC_DELTA)
            
            action_hist = jnp.concatenate([action_hist[:, 1:], action[:, None]], axis=1)
            exo_hist = jnp.concatenate([exo_hist[:, 1:, :], exo_row[:, None, :3]], axis=1)
            lataccel_hist = jnp.concatenate([lataccel_hist[:, 1:], next_lataccel[:, None]], axis=1)
            
            new_carry = (action_hist, lataccel_hist, exo_hist, next_lataccel)
            return new_carry, next_lataccel
        return step

    # =========================================================================
    # Controller implementations (all batched)
    # =========================================================================
    
    def generate_pid_actions(exo_data, init_lataccel, p, i, d):
        """Generate PID actions for entire trajectory. Batched."""
        # exo_data: [horizon, batch, 4], init_lataccel: [batch]
        horizon, batch_size, _ = exo_data.shape
        targets = exo_data[:, :, 3]  # [horizon, batch]
        
        def pid_step(carry, target_t):
            lataccel, error_integral, prev_error = carry
            error = target_t - lataccel
            error_integral_new = error_integral + error
            error_diff = error - prev_error
            action = p * error + i * error_integral_new + d * error_diff
            action = jnp.clip(action, -2, 2)
            # Approximate next lataccel (won't be exact but good enough for action generation)
            next_lataccel = lataccel + 0.3 * (target_t - lataccel)  # Simple lag approx
            return (next_lataccel, error_integral_new, error), action
        
        init_carry = (init_lataccel, jnp.zeros(batch_size), jnp.zeros(batch_size))
        _, actions = lax.scan(pid_step, init_carry, targets)
        return actions  # [horizon, batch]
    
    def generate_smooth_random_actions(key, horizon, batch_size, smoothing=0.9, scale=0.5):
        """Generate smoothed random walk actions. Batched."""
        noise = jax.random.normal(key, (horizon, batch_size)) * scale
        
        def smooth_step(prev_action, noise_t):
            action = smoothing * prev_action + (1 - smoothing) * noise_t
            action = jnp.clip(action, -1.5, 1.5)
            return action, action
        
        _, actions = lax.scan(smooth_step, jnp.zeros(batch_size), noise)
        return actions  # [horizon, batch]
    
    def generate_sine_actions(horizon, batch_size, amplitude, freq, key):
        """Generate sinusoidal actions with random phase. Batched."""
        t = jnp.arange(horizon)[:, None]  # [horizon, 1]
        phases = jax.random.uniform(key, (batch_size,)) * 2 * jnp.pi  # [batch]
        actions = amplitude * jnp.sin(2 * jnp.pi * freq * t + phases)  # [horizon, batch]
        return actions
    
    def generate_scaled_target_actions(exo_data, scale):
        """Generate actions that track target with scaling. Batched."""
        targets = exo_data[:, :, 3]  # [horizon, batch]
        actions = jnp.clip(scale * targets, -1.5, 1.5)
        return actions

    # =========================================================================
    # Run simulation with pre-generated actions
    # =========================================================================
    
    @eqx.filter_jit
    def run_with_actions(init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, actions):
        """Run simulation with given action sequence."""
        init_carry = (init_action_hist, init_lataccel_hist, init_exo_hist, init_lataccel_hist[:, -1])
        step_fn = make_sim_step(model)
        _, lataccels = lax.scan(step_fn, init_carry, (exo_data, actions))
        return lataccels  # [horizon, batch]

    # =========================================================================
    # Warmup
    # =========================================================================
    
    @eqx.filter_jit
    def run_warmup(init_action_hist, init_lataccel_hist, init_exo_hist, warmup_exos, warmup_actions):
        outputs = run_simulation(
            model, init_action_hist, init_lataccel_hist, init_exo_hist,
            warmup_exos, warmup_actions
        )
        post_action_hist = outputs[-CONTEXT_LENGTH:, :, 1].T
        post_lataccel_hist = outputs[-CONTEXT_LENGTH:, :, 0].T
        post_exo_hist = outputs[-CONTEXT_LENGTH:, :, 2:5].transpose(1, 0, 2)
        init_lataccel = post_lataccel_hist[:, -1]
        return post_action_hist, post_lataccel_hist, post_exo_hist, init_lataccel

    # =========================================================================
    # Compile
    # =========================================================================
    
    print("Compiling...")
    warmup_len = CONTROL_START_IDX - CONTEXT_LENGTH
    dummy_hist = jnp.zeros((batch_size, CONTEXT_LENGTH), dtype=jnp.float32)
    dummy_exo_hist = jnp.zeros((batch_size, CONTEXT_LENGTH, 3), dtype=jnp.float32)
    dummy_warmup_exo = jnp.zeros((warmup_len, batch_size, 4), dtype=jnp.float32)
    dummy_warmup_act = jnp.zeros((warmup_len, batch_size), dtype=jnp.float32)
    dummy_pid_exo = jnp.zeros((horizon, batch_size, 4), dtype=jnp.float32)
    dummy_actions = jnp.zeros((horizon, batch_size), dtype=jnp.float32)
    
    _ = run_warmup(dummy_hist, dummy_hist, dummy_exo_hist, dummy_warmup_exo, dummy_warmup_act)
    _ = run_with_actions(dummy_hist, dummy_hist, dummy_exo_hist, dummy_pid_exo, dummy_actions)
    print("Compilation done.")

    # =========================================================================
    # Collection
    # =========================================================================
    
    n_batches = (n_trajectories + batch_size - 1) // batch_size
    all_outputs = []
    all_init_lataccels = []
    key = jax.random.PRNGKey(seed)

    total_start = time.perf_counter()
    
    controller_types = ['pid', 'lazy_pid', 'smooth_random', 'sine', 'scaled_target']
    ctrl_counts = {c: 0 for c in controller_types}

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        current_batch_size = min(batch_size, n_trajectories - batch_start)

        key, key_files, key_ctrl = jax.random.split(key, 3)

        file_indices = np.array(jax.random.randint(key_files, (current_batch_size,), 0, n_files))
        batch_driving = driving_data[file_indices]

        init_action_hist = jnp.array(batch_driving[:, :CONTEXT_LENGTH, 4])
        init_lataccel_hist = jnp.array(batch_driving[:, :CONTEXT_LENGTH, 3])
        init_exo_hist = jnp.array(batch_driving[:, :CONTEXT_LENGTH, :3])
        warmup_exos = jnp.array(batch_driving[:, CONTEXT_LENGTH:CONTROL_START_IDX, :4]).transpose(1, 0, 2)
        warmup_actions = jnp.array(batch_driving[:, CONTEXT_LENGTH:CONTROL_START_IDX, 4]).T
        pid_exos = jnp.array(batch_driving[:, CONTROL_START_IDX:CONTROL_START_IDX+horizon, :4]).transpose(1, 0, 2)

        post_action, post_lataccel, post_exo, init_lat = run_warmup(
            init_action_hist, init_lataccel_hist, init_exo_hist,
            warmup_exos, warmup_actions
        )

        ctrl_type = controller_types[batch_idx % len(controller_types)]
        ctrl_counts[ctrl_type] += 1
        
        start = time.perf_counter()
        
        if ctrl_type == 'pid':
            p = float(np.random.uniform(0.10, 0.30))
            i = float(np.random.uniform(0.05, 0.15))
            d = float(np.random.uniform(-0.10, -0.02))
            actions = generate_pid_actions(pid_exos, init_lat, p, i, d)
            desc = f"PID({p:.2f},{i:.2f},{d:.2f})"
            
        elif ctrl_type == 'lazy_pid':
            p = float(np.random.uniform(0.03, 0.10))
            i = float(np.random.uniform(0.01, 0.05))
            d = float(np.random.uniform(-0.04, -0.01))
            actions = generate_pid_actions(pid_exos, init_lat, p, i, d)
            desc = f"LazyPID({p:.2f},{i:.2f},{d:.2f})"
            
        elif ctrl_type == 'smooth_random':
            smoothing = float(np.random.uniform(0.85, 0.95))
            scale = float(np.random.uniform(0.2, 0.5))
            actions = generate_smooth_random_actions(key_ctrl, horizon, current_batch_size, smoothing, scale)
            desc = f"SmoothRand(s={smoothing:.2f},sc={scale:.2f})"
            
        elif ctrl_type == 'sine':
            amplitude = float(np.random.uniform(0.2, 0.6))
            freq = float(np.random.uniform(0.02, 0.08))
            actions = generate_sine_actions(horizon, current_batch_size, amplitude, freq, key_ctrl)
            desc = f"Sine(A={amplitude:.2f},f={freq:.2f})"
            
        elif ctrl_type == 'scaled_target':
            scale = float(np.random.uniform(0.15, 0.40))
            actions = generate_scaled_target_actions(pid_exos, scale)
            desc = f"ScaledTarget(k={scale:.2f})"
        
        # Run simulation with generated actions
        lataccels = run_with_actions(post_action, post_lataccel, post_exo, pid_exos, actions)
        jax.block_until_ready(lataccels)
        elapsed = time.perf_counter() - start

        # Combine: [horizon, batch, 6] = (lataccel, action, roll, v, a, target)
        full_output = jnp.stack([
            lataccels,
            actions,
            pid_exos[:, :, 0],
            pid_exos[:, :, 1],
            pid_exos[:, :, 2],
            pid_exos[:, :, 3],
        ], axis=-1)

        samples = current_batch_size * horizon
        print(f"Batch {batch_idx+1}/{n_batches}: {samples:,} in {elapsed:.2f}s ({samples/elapsed:,.0f}/s) {desc}")

        all_outputs.append(np.array(full_output))
        all_init_lataccels.append(np.array(init_lat))

    total_elapsed = time.perf_counter() - total_start
    total_samples = n_trajectories * horizon
    print(f"\nTotal: {total_samples:,} samples in {total_elapsed:.1f}s ({total_samples/total_elapsed:,.0f}/s)")
    print(f"Controller distribution: {ctrl_counts}")

    # Combine
    combined_outputs = np.concatenate(all_outputs, axis=1)[:, :n_trajectories, :]
    combined_outputs = np.transpose(combined_outputs, (1, 0, 2))
    combined_init_lataccels = np.concatenate(all_init_lataccels)[:n_trajectories]

    print(f"Output shapes: trajectories={combined_outputs.shape}, init_lataccels={combined_init_lataccels.shape}")
    
    lataccels = combined_outputs[:, :, 0]
    actions = combined_outputs[:, :, 1]
    print(f"Lataccel range: [{lataccels.min():.2f}, {lataccels.max():.2f}]")
    print(f"Action range: [{actions.min():.2f}, {actions.max():.2f}]")
    
    # Saturation stats
    lat_sat = np.any((lataccels == -5.0) | (lataccels == 5.0), axis=1)
    act_sat = np.any((actions == -2.0) | (actions == 2.0), axis=1)
    print(f"Lataccel saturated: {lat_sat.sum()} ({lat_sat.mean()*100:.1f}%)")
    print(f"Action saturated: {act_sat.sum()} ({act_sat.mean()*100:.1f}%)")
    print(f"Either saturated: {(lat_sat | act_sat).sum()} ({(lat_sat | act_sat).mean()*100:.1f}%)")

    # Save
    print("Compressing...")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        trajectories=combined_outputs,
        init_lataccels=combined_init_lataccels,
        n_trajectories=n_trajectories,
        horizon=horizon,
    )
    
    data_bytes = buffer.getvalue()
    print(f"Compressed size: {len(data_bytes) / 1024 / 1024:.1f} MB")
    
    return data_bytes


@app.local_entrypoint()
def main(
    n_trajectories: int = 100000,
    horizon: int = 400,
    batch_size: int = 2000,
    output_path: str = "bicycle_training_diverse.npz",
):
    """Collect diverse training data for bicycle model."""
    import time

    print(f"Collecting {n_trajectories} trajectories, horizon={horizon}")
    print(f"Batch size: {batch_size}")

    start = time.perf_counter()
    data_bytes = collect_training_data.remote(
        n_trajectories=n_trajectories,
        horizon=horizon,
        batch_size=batch_size,
        seed=42,
    )
    elapsed = time.perf_counter() - start

    with open(output_path, "wb") as f:
        f.write(data_bytes)

    print(f"\n=== Results ===")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Saved to {output_path} ({len(data_bytes)/1024/1024:.1f} MB)")
