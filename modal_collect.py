"""
Modal script for massive data collection on H100 GPUs.

Uses real driving data from the data/ folder.
Data is persisted in a Modal Volume to avoid re-uploading.

Usage:
    # First time: upload data
    modal run modal_collect.py --upload-data

    # Then collect:
    modal run modal_collect.py --n-trajectories 1000
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
def collect_all(n_trajectories: int, horizon: int, batch_size: int, seed: int) -> bytes:
    """Collect all trajectories on a single GPU using real driving data."""
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import time
    import io

    from tinyphysics_eqx import create_model, run_simulation, CONTEXT_LENGTH

    print(f"JAX devices: {jax.devices()}")
    print(f"Collecting {n_trajectories} trajectories, horizon={horizon}, batch_size={batch_size}")

    # Load real driving data from volume
    # Shape: [n_files, n_steps, 5] - columns: roll_lataccel, v_ego, a_ego, target, steer_command
    driving_data = np.load("/data/driving_data.npy")
    n_files, max_steps, _ = driving_data.shape
    print(f"Loaded driving data: {driving_data.shape}")

    # Load model
    model = create_model("/root/models/tinyphysics.onnx")

    # JIT compile
    @eqx.filter_jit
    def run_sim(init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, actions):
        return run_simulation(model, init_action_hist, init_lataccel_hist, init_exo_hist, exo_data, actions)

    def sample_actions(key, shape):
        keys = jax.random.split(key, 5)
        r = jax.random.uniform(keys[0], shape)
        uniform = jax.random.uniform(keys[1], shape, minval=-2, maxval=2)
        normal = jnp.clip(jax.random.normal(keys[2], shape) * 0.5, -2, 2)
        grid = jnp.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])[jax.random.randint(keys[3], shape, 0, 7)]
        extremes = jnp.array([-2.0, 2.0])[jax.random.randint(keys[4], shape, 0, 2)]
        return jnp.where(r < 0.4, uniform, jnp.where(r < 0.7, normal, jnp.where(r < 0.9, grid, extremes)))

    # Warmup with full batch size to trigger JIT + CUDA autotuning
    print("Compiling (full batch size for proper warmup)...")
    dummy_hist = jnp.zeros((batch_size, CONTEXT_LENGTH), dtype=jnp.float32)
    dummy_exo_hist = jnp.zeros((batch_size, CONTEXT_LENGTH, 3), dtype=jnp.float32)  # [batch, 20, 3] = roll, v, a
    dummy_exo = jnp.zeros((horizon, batch_size, 4), dtype=jnp.float32)
    dummy_actions = jnp.zeros((horizon, batch_size), dtype=jnp.float32)
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

        # Initialize from first CONTEXT_LENGTH steps of real data
        init_lataccel_hist = jnp.array(batch_driving[:, :CONTEXT_LENGTH, 3])  # target as initial lataccel
        init_action_hist = jnp.array(batch_driving[:, :CONTEXT_LENGTH, 4])    # real steer commands
        init_exo_hist = jnp.array(batch_driving[:, :CONTEXT_LENGTH, :3])      # [batch, 20, 3] = roll, v, a

        # Exogenous data for simulation: [horizon, batch, 4] = roll, v, a, target
        # Start from CONTEXT_LENGTH, run for horizon steps
        exo_start = CONTEXT_LENGTH
        exo_end = min(exo_start + horizon, max_steps)
        actual_horizon = exo_end - exo_start

        exo_batch = batch_driving[:, exo_start:exo_end, :4]  # [batch, horizon, 4]
        exo_data = jnp.array(exo_batch.transpose(1, 0, 2))   # [horizon, batch, 4]

        # Sample random actions for exploration
        actions = sample_actions(keys[1], (actual_horizon, current_batch_size))

        # Run
        start = time.perf_counter()
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
    # Stack along batch dimension
    combined = np.concatenate(all_outputs, axis=1)  # [horizon, total_trajectories, 6]
    combined = combined[:, :n_trajectories, :]  # Trim to exact count

    # Transpose to [n_trajectories, horizon, 6] for easier access
    combined = np.transpose(combined, (1, 0, 2))

    print(f"Output shape: {combined.shape}")
    print(f"Compressing and saving...")

    # Save to bytes
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        data=combined,  # [n_trajectories, horizon, 6] - columns: lataccel, action, roll, v, a, target
        n_trajectories=n_trajectories,
        horizon=horizon,
    )

    data_bytes = buffer.getvalue()
    print(f"Compressed size: {len(data_bytes) / 1024 / 1024:.1f} MB")

    return data_bytes


@app.local_entrypoint()
def main(
    upload_data_flag: bool = False,
    n_trajectories: int = 100,  # Small default for debugging
    horizon: int = 400,
    batch_size: int = 500,
):
    """
    Collect data on a single H100 and download the result.

    Usage:
        # First time: upload preprocessed data
        modal run modal_collect.py --upload-data-flag

        # Then collect (small for debugging):
        modal run modal_collect.py --n-trajectories 100

        # Production run:
        modal run modal_collect.py --n-trajectories 10000 --batch-size 2000
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
    print(f"Expected samples: {n_trajectories * horizon:,}")

    start = time.perf_counter()

    # Run on GPU
    data_bytes = collect_all.remote(
        n_trajectories=n_trajectories,
        horizon=horizon,
        batch_size=batch_size,
        seed=42,
    )

    elapsed = time.perf_counter() - start

    # Save locally
    output_path = "koopman_data_modal.npz"
    with open(output_path, "wb") as f:
        f.write(data_bytes)

    total_samples = n_trajectories * horizon
    print(f"\n=== Results ===")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Throughput: {total_samples/elapsed:,.0f} samples/sec (including transfer)")
    print(f"Saved to {output_path} ({len(data_bytes)/1024/1024:.1f} MB)")

    # Quick stats (import numpy only after Modal work is done)
    import numpy as np
    data = np.load(output_path)['data']
    lataccels = data[:, :, 0]
    print(f"\n=== Data Quality ===")
    print(f"Lataccel range: [{lataccels.min():.2f}, {lataccels.max():.2f}]")
    print(f"Lataccel mean: {lataccels.mean():.3f}")
    print(f"Lataccel std: {lataccels.std():.3f}")
    print(f"Saturated at -5: {(lataccels == -5.0).mean()*100:.1f}%")
    print(f"Saturated at +5: {(lataccels == 5.0).mean()*100:.1f}%")
