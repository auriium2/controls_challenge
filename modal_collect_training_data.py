"""
Modal script for collecting bicycle training data on H100 GPU.

Generates (actions, transformer_lataccels, init_lataccel, exo_data) tuples
that can be used to train the bicycle model offline.

Usage:
    modal run modal_collect_training_data.py --n-trajectories 50000
"""

import modal

app = modal.App("bicycle-training-data")

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
    """Collect training data for bicycle model."""
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import time
    import io

    from tinyphysics_eqx import (
        create_model, run_simulation, run_simulation_pid, 
        CONTEXT_LENGTH
    )

    CONTROL_START_IDX = 100
    
    # PID gain ranges (safe region to avoid saturation)
    PID_RANGES = {
        'p': (0.10, 0.30),
        'i': (0.05, 0.15),
        'd': (-0.10, -0.02),
    }

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

    # Load model
    model = create_model("/root/models/tinyphysics.onnx")

    # JIT compile simulation functions
    @eqx.filter_jit
    def run_warmup_and_pid(init_action_hist, init_lataccel_hist, init_exo_hist,
                           warmup_exos, warmup_actions, pid_exos, pid_p, pid_i, pid_d):
        """Run warmup + PID in one go, return everything needed for training."""
        # Warmup phase
        warmup_outputs = run_simulation(
            model, init_action_hist, init_lataccel_hist, init_exo_hist,
            warmup_exos, warmup_actions
        )
        # warmup_outputs: [80, batch, 6]
        
        # Extract post-warmup histories
        post_action_hist = warmup_outputs[-CONTEXT_LENGTH:, :, 1].T  # [batch, 20]
        post_lataccel_hist = warmup_outputs[-CONTEXT_LENGTH:, :, 0].T
        post_exo_hist = warmup_outputs[-CONTEXT_LENGTH:, :, 2:5].transpose(1, 0, 2)  # [batch, 20, 3]
        
        # PID phase
        pid_outputs = run_simulation_pid(
            model, post_action_hist, post_lataccel_hist, post_exo_hist, pid_exos,
            p=pid_p, i=pid_i, d=pid_d
        )
        # pid_outputs: [horizon, batch, 6] = (lataccel, action, roll, v, a, target)
        
        # Init lataccel for bicycle (last value before PID starts)
        init_lataccel = post_lataccel_hist[:, -1]  # [batch]
        
        return pid_outputs, init_lataccel

    # Warmup compilation
    print("Compiling...")
    warmup_len = CONTROL_START_IDX - CONTEXT_LENGTH  # 80
    dummy_hist = jnp.zeros((batch_size, CONTEXT_LENGTH), dtype=jnp.float32)
    dummy_exo_hist = jnp.zeros((batch_size, CONTEXT_LENGTH, 3), dtype=jnp.float32)
    dummy_warmup_exo = jnp.zeros((warmup_len, batch_size, 4), dtype=jnp.float32)
    dummy_warmup_act = jnp.zeros((warmup_len, batch_size), dtype=jnp.float32)
    dummy_pid_exo = jnp.zeros((horizon, batch_size, 4), dtype=jnp.float32)
    
    _ = run_warmup_and_pid(
        dummy_hist, dummy_hist, dummy_exo_hist,
        dummy_warmup_exo, dummy_warmup_act, dummy_pid_exo,
        0.195, 0.1, -0.053
    )
    print("Compilation done.")

    # Collect data
    n_batches = (n_trajectories + batch_size - 1) // batch_size
    all_outputs = []
    all_init_lataccels = []
    key = jax.random.PRNGKey(seed)

    total_start = time.perf_counter()
    warmup_len = CONTROL_START_IDX - CONTEXT_LENGTH

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        current_batch_size = min(batch_size, n_trajectories - batch_start)

        key, key_files, key_pid = jax.random.split(key, 3)

        # Sample random files
        file_indices = np.array(jax.random.randint(key_files, (current_batch_size,), 0, n_files))
        batch_driving = driving_data[file_indices]  # [batch, steps, 5]

        # Prepare data
        # Initial histories (steps 0-19)
        init_action_hist = jnp.array(batch_driving[:, :CONTEXT_LENGTH, 4])
        init_lataccel_hist = jnp.array(batch_driving[:, :CONTEXT_LENGTH, 3])
        init_exo_hist = jnp.array(batch_driving[:, :CONTEXT_LENGTH, :3])

        # Warmup (steps 20-99)
        warmup_exos = jnp.array(batch_driving[:, CONTEXT_LENGTH:CONTROL_START_IDX, :4]).transpose(1, 0, 2)
        warmup_actions = jnp.array(batch_driving[:, CONTEXT_LENGTH:CONTROL_START_IDX, 4]).T

        # PID phase (steps 100+)
        pid_exos = jnp.array(batch_driving[:, CONTROL_START_IDX:CONTROL_START_IDX+horizon, :4]).transpose(1, 0, 2)

        # Sample random PID gains for this batch
        pid_p = float(np.random.uniform(*PID_RANGES['p']))
        pid_i = float(np.random.uniform(*PID_RANGES['i']))
        pid_d = float(np.random.uniform(*PID_RANGES['d']))

        # Run simulation
        start = time.perf_counter()
        pid_outputs, init_lataccel = run_warmup_and_pid(
            init_action_hist, init_lataccel_hist, init_exo_hist,
            warmup_exos, warmup_actions, pid_exos,
            pid_p, pid_i, pid_d
        )
        jax.block_until_ready(pid_outputs)
        elapsed = time.perf_counter() - start

        samples = current_batch_size * horizon
        print(f"Batch {batch_idx+1}/{n_batches}: {samples:,} samples in {elapsed:.2f}s "
              f"({samples/elapsed:,.0f}/s) PID=({pid_p:.3f},{pid_i:.3f},{pid_d:.3f})")

        all_outputs.append(np.array(pid_outputs))
        all_init_lataccels.append(np.array(init_lataccel))

    total_elapsed = time.perf_counter() - total_start
    total_samples = n_trajectories * horizon
    print(f"\nTotal: {total_samples:,} samples in {total_elapsed:.1f}s ({total_samples/total_elapsed:,.0f}/s)")

    # Combine outputs
    # pid_outputs: [horizon, batch, 6] -> concatenate along batch axis
    combined_outputs = np.concatenate(all_outputs, axis=1)[:, :n_trajectories, :]
    combined_outputs = np.transpose(combined_outputs, (1, 0, 2))  # [n_traj, horizon, 6]
    
    combined_init_lataccels = np.concatenate(all_init_lataccels)[:n_trajectories]  # [n_traj]

    print(f"Output shapes: trajectories={combined_outputs.shape}, init_lataccels={combined_init_lataccels.shape}")
    
    # Data format:
    # combined_outputs[:, :, 0] = transformer_lataccels
    # combined_outputs[:, :, 1] = pid_actions
    # combined_outputs[:, :, 2:6] = exo_data (roll, v, a, target)
    # combined_init_lataccels = init lataccel for bicycle

    # Quality check
    lataccels = combined_outputs[:, :, 0]
    actions = combined_outputs[:, :, 1]
    print(f"Lataccel range: [{lataccels.min():.2f}, {lataccels.max():.2f}]")
    print(f"Action range: [{actions.min():.2f}, {actions.max():.2f}]")

    # Save
    print("Compressing...")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        trajectories=combined_outputs,  # [n_traj, horizon, 6]
        init_lataccels=combined_init_lataccels,  # [n_traj]
        n_trajectories=n_trajectories,
        horizon=horizon,
    )
    
    data_bytes = buffer.getvalue()
    print(f"Compressed size: {len(data_bytes) / 1024 / 1024:.1f} MB")
    
    return data_bytes


@app.local_entrypoint()
def main(
    n_trajectories: int = 10000,
    horizon: int = 400,
    batch_size: int = 1000,
    output_path: str = "bicycle_training_data.npz",
):
    """Collect training data for bicycle model."""
    import time

    print(f"Collecting {n_trajectories} trajectories, horizon={horizon}")
    print(f"Batch size: {batch_size}")
    print(f"Expected samples: {n_trajectories * horizon:,}")

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
