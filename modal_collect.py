"""
Modal script for massive data collection on H100 GPUs.

Uses real driving data from the data/ folder.
Data is persisted in a Modal Volume to avoid re-uploading.

Usage:
    # First time: upload data
    modal run modal_collect.py --upload-data-flag

    # Then collect with random actions (original):
    modal run modal_collect.py --n-trajectories 1000

    # Collect with PID controller (better action diversity):
    modal run modal_collect.py --n-trajectories 1000 --use-pid
"""

import modal

app = modal.App("koopman-data-collection")

# Volume to persist data between runs
volume = modal.Volume.from_name("koopman-data", create_if_missing=True)

# Create image with JAX + CUDA and local files
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


@app.function(image=image, timeout=120, volumes={"/data": volume})
def upload_data(data_bytes: bytes):
    """Upload preprocessed data to the volume."""
    import numpy as np
    import io

    # Load and save to volume
    data = np.load(io.BytesIO(data_bytes))['data']
    np.save("/data/driving_data.npy", data)
    volume.commit()  # Persist to volume
    print(f"Uploaded data with shape {data.shape} to volume")
    return list(data.shape)  # Return list for JSON serialization


@app.function(
    image=image,
    gpu="H100",
    timeout=1200,
    volumes={"/data": volume},
)
def collect_all(n_trajectories: int, horizon: int, batch_size: int, seed: int, use_pid: bool = False, noise_std: float = 0.3) -> bytes:
    """Collect all trajectories on a single GPU using real driving data."""
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import time
    import io

    from tinyphysics_eqx import create_model, run_simulation, run_simulation_pid, run_simulation_noisy_pid, CONTEXT_LENGTH

    # Match original tinyphysics.py constants
    CONTROL_START_IDX = 100  # Controller takes over at step 100

    print(f"JAX devices: {jax.devices()}")
    print(f"Collecting {n_trajectories} trajectories, horizon={horizon}, batch_size={batch_size}")
    print(f"Action mode: {'PID controller' if use_pid else 'random sampling'}")

    # Load real driving data from volume
    # Shape: [n_files, n_steps, 5] - columns: roll_lataccel, v_ego, a_ego, target, steer_command
    driving_data = np.load("/data/driving_data.npy")
    n_files, max_steps, _ = driving_data.shape
    print(f"Loaded driving data: {driving_data.shape}")

    # Validate horizon fits in data (we start from CONTROL_START_IDX=100)
    available_horizon = max_steps - CONTROL_START_IDX
    if horizon > available_horizon:
        print(f"Warning: requested horizon {horizon} > available {available_horizon}, using {available_horizon}")
        horizon = available_horizon

    # Load model
    model = create_model("/root/models/tinyphysics.onnx")

    # JIT compile
    @eqx.filter_jit
    def run_sim(init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, actions):
        return run_simulation(model, init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, actions)

    @eqx.filter_jit
    def run_sim_pid(init_action_hist, init_lataccel_hist, init_exo_hist, exo_data):
        return run_simulation_pid(model, init_action_hist, init_lataccel_hist, init_exo_hist, exo_data)

    def run_sim_noisy_pid(init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, key):
        return run_simulation_noisy_pid(model, init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, key, noise_std=noise_std)

    def sample_actions(key, shape, driving_batch):
        """
        Sample actions from real driving distribution (steps 0-99 have valid steer commands).
        
        Real driving: mean~0, std~0.26, rarely exceeds |1.0|
        """
        horizon, batch_size = shape
        keys = jax.random.split(key, 2)
        
        # Get real steer commands from steps 0-99 (before controller takes over)
        # driving_batch: [batch, steps, 5], column 4 is steer_command
        real_actions = driving_batch[:, :CONTROL_START_IDX, 4]  # [batch, 100]
        
        # Randomly sample timesteps from the valid range for each (horizon, batch) position
        # This gives us realistic action values
        time_indices = jax.random.randint(keys[0], (horizon, batch_size), 0, CONTROL_START_IDX)
        batch_indices = jnp.arange(batch_size)[None, :].repeat(horizon, axis=0)
        
        sampled_actions = real_actions[batch_indices, time_indices]  # [horizon, batch]
        
        # Add small exploration noise
        noise = jax.random.normal(keys[1], shape) * 0.1
        actions = sampled_actions + noise
        
        # Clip to valid range
        return jnp.clip(actions, -2, 2)

    # Warmup with full batch size to trigger JIT + CUDA autotuning
    print("Compiling (full batch size for proper warmup)...")
    dummy_hist = jnp.zeros((batch_size, CONTEXT_LENGTH), dtype=jnp.float32)
    dummy_exo_hist = jnp.zeros((batch_size, CONTEXT_LENGTH, 3), dtype=jnp.float32)
    dummy_exo = jnp.zeros((horizon, batch_size, 4), dtype=jnp.float32)
    dummy_actions = jnp.zeros((horizon, batch_size), dtype=jnp.float32)
    if use_pid:
        _ = run_sim_pid(dummy_hist, dummy_hist, dummy_exo_hist, dummy_exo)
    else:
        _ = run_sim(dummy_hist, dummy_hist, dummy_exo_hist, dummy_exo, dummy_actions)
    print("Compilation done.")

    # Collect in batches
    n_batches = (n_trajectories + batch_size - 1) // batch_size
    all_outputs = []
    key = jax.random.PRNGKey(seed)

    total_start = time.perf_counter()
    total_samples = 0

    for batch_idx in range(n_batches):
        batch_start_idx = batch_idx * batch_size
        current_batch_size = min(batch_size, n_trajectories - batch_start_idx)

        key, *keys = jax.random.split(key, 3)

        # Select random driving files for this batch
        file_indices = jax.random.randint(keys[0], (current_batch_size,), 0, n_files)
        file_indices = np.array(file_indices)

        # Extract data for selected files
        # driving_data columns: roll_lataccel, v_ego, a_ego, target, steer_command
        batch_driving = driving_data[file_indices]  # [batch, steps, 5]

        # Initialize from steps leading up to CONTROL_START_IDX (steps 80-99)
        # This matches where the controller would take over in the original simulator
        init_start = CONTROL_START_IDX - CONTEXT_LENGTH  # step 80
        init_end = CONTROL_START_IDX  # step 100
        
        init_lataccel_hist = jnp.array(batch_driving[:, init_start:init_end, 3])  # target as initial lataccel
        init_action_hist = jnp.array(batch_driving[:, init_start:init_end, 4])    # real steer commands
        init_exo_hist = jnp.array(batch_driving[:, init_start:init_end, :3])      # [batch, 20, 3] = roll, v, a

        # Exogenous data for simulation: [horizon, batch, 4] = roll, v, a, target
        # Start from CONTROL_START_IDX, run for horizon steps
        exo_start = CONTROL_START_IDX
        exo_end = exo_start + horizon

        exo_batch = batch_driving[:, exo_start:exo_end, :4]  # [batch, horizon, 4]
        exo_data = jnp.array(exo_batch.transpose(1, 0, 2))   # [horizon, batch, 4]

        # Run simulation
        start = time.perf_counter()
        if use_pid:
            # Use noisy PID controller for action selection (adds exploration)
            outputs = run_sim_noisy_pid(init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, keys[1])
            outputs = jnp.array(outputs)  # Ensure it's a JAX array for block_until_ready
        else:
            # Sample actions from real driving distribution
            actions = sample_actions(keys[1], (horizon, current_batch_size), batch_driving)
            outputs = run_sim(init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, actions)
        outputs.block_until_ready()
        elapsed = time.perf_counter() - start

        samples = current_batch_size * horizon
        total_samples += samples
        print(f"Batch {batch_idx+1}/{n_batches}: {samples:,} samples in {elapsed:.2f}s ({samples/elapsed:,.0f} samples/sec)")

        all_outputs.append(np.array(outputs))

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal: {total_samples:,} samples in {total_elapsed:.1f}s ({total_samples/total_elapsed:,.0f} samples/sec)")

    # Concatenate all outputs: each is [horizon, batch, 6]
    combined = np.concatenate(all_outputs, axis=1)  # [horizon, total_trajectories, 6]
    combined = combined[:, :n_trajectories, :]  # Trim to exact count

    # Transpose to [n_trajectories, horizon, 6] for easier access
    combined = np.transpose(combined, (1, 0, 2))

    print(f"Output shape: {combined.shape}")
    print(f"Output columns: lataccel, action, roll, v, a, target")

    # Quick quality check
    lataccels = combined[:, :, 0]
    print(f"Lataccel range: [{lataccels.min():.2f}, {lataccels.max():.2f}]")
    print(f"Saturated: {(np.abs(lataccels) >= 4.99).mean()*100:.1f}%")

    print(f"Compressing...")

    # Save to bytes
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        data=combined,  # [n_trajectories, horizon, 6]
        n_trajectories=n_trajectories,
        horizon=horizon,
    )

    data_bytes = buffer.getvalue()
    print(f"Compressed size: {len(data_bytes) / 1024 / 1024:.1f} MB")

    return data_bytes


@app.local_entrypoint()
def main(
    upload_data_flag: bool = False,
    n_trajectories: int = 100,
    horizon: int = 490,  # Max is 498 (steps 100-597), use 490 for safety
    batch_size: int = 500,
    use_pid: bool = False,
    noise_std: float = 0.3,
    output_path: str = "koopman_data.npz",
):
    """
    Collect data on a single H100 and download the result.

    Usage:
        # First time: upload preprocessed data
        modal run modal_collect.py --upload-data-flag

        # Then collect (small for debugging):
        modal run modal_collect.py --n-trajectories 100

        # Production run with random actions:
        modal run modal_collect.py --n-trajectories 10000 --batch-size 2000

        # Production run with noisy PID controller:
        modal run modal_collect.py --n-trajectories 10000 --batch-size 2000 --use-pid --output-path koopman_data_pid.npz

        # Adjust exploration noise (default 0.3):
        modal run modal_collect.py --n-trajectories 10000 --use-pid --noise-std 0.5
    """
    import time

    if upload_data_flag:
        # Upload preprocessed data to volume
        print("Uploading preprocessed data to Modal volume...")
        with open("data_preprocessed.npz", "rb") as f:
            data_bytes = f.read()

        shape = upload_data.remote(data_bytes)
        print(f"Uploaded data with shape {shape}")
        return

    print(f"Collecting {n_trajectories} trajectories with horizon {horizon}")
    print(f"Batch size: {batch_size}")
    print(f"Action mode: {'noisy PID (noise_std=' + str(noise_std) + ')' if use_pid else 'random sampling'}")
    print(f"Expected samples: {n_trajectories * horizon:,}")

    start = time.perf_counter()

    # Run on GPU
    data_bytes = collect_all.remote(
        n_trajectories=n_trajectories,
        horizon=horizon,
        batch_size=batch_size,
        seed=42,
        use_pid=use_pid,
        noise_std=noise_std,
    )

    elapsed = time.perf_counter() - start

    # Save locally
    with open(output_path, "wb") as f:
        f.write(data_bytes)

    total_samples = n_trajectories * horizon
    print(f"\n=== Results ===")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Throughput: {total_samples/elapsed:,.0f} samples/sec (including transfer)")
    print(f"Saved to {output_path} ({len(data_bytes)/1024/1024:.1f} MB)")

    # Quick stats
    import numpy as np
    data = np.load(output_path)['data']
    lataccels = data[:, :, 0]
    print(f"\n=== Data Quality ===")
    print(f"Shape: {data.shape}")
    print(f"Lataccel range: [{lataccels.min():.2f}, {lataccels.max():.2f}]")
    print(f"Lataccel mean: {lataccels.mean():.3f}")
    print(f"Lataccel std: {lataccels.std():.3f}")
    print(f"Saturated at -5: {(lataccels == -5.0).mean()*100:.1f}%")
    print(f"Saturated at +5: {(lataccels == 5.0).mean()*100:.1f}%")
