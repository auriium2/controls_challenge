"""
Modal script for training bicycle model on H100 GPU.

Usage:
    modal run modal_train_bicycle.py --epochs 30 --num-files 10000
"""

import modal

app = modal.App("bicycle-training")

# Volume to persist data between runs
volume = modal.Volume.from_name("koopman-data", create_if_missing=True)

# Create image with JAX + CUDA and local files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]==0.6.0",
        "equinox>=0.11.0",
        "optax>=0.2.0",
        "onnx>=1.15.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    )
    .add_local_file("models/tinyphysics.onnx", remote_path="/root/models/tinyphysics.onnx")
    .add_local_file("tinyphysics_eqx.py", remote_path="/root/tinyphysics_eqx.py")
    .add_local_file("bicycle_model.py", remote_path="/root/bicycle_model.py")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/data": volume},
)
def train_bicycle_remote(
    epochs: int,
    batch_size: int,
    horizon: int,
    learning_rate: float,
    num_files: int,
) -> bytes:
    """Train bicycle model on GPU."""
    import sys
    sys.path.insert(0, "/root")
    
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import numpy as np
    import optax
    from tqdm import tqdm
    import io
    
    from bicycle_model import BicycleModel, rollout_bicycle
    from tinyphysics_eqx import (
        create_model, run_simulation_pid, run_simulation, 
        CONTEXT_LENGTH
    )
    
    print(f"JAX devices: {jax.devices()}")
    
    ACC_G = 9.81
    DT = 0.1
    CONTROL_START_IDX = 100
    
    # Baseline PID gains and safe variation ranges
    PID_RANGES = {
        'p': (0.10, 0.30),
        'i': (0.05, 0.15),
        'd': (-0.10, -0.02),
    }
    
    def sample_pid_gains():
        return {
            'p': np.random.uniform(*PID_RANGES['p']),
            'i': np.random.uniform(*PID_RANGES['i']),
            'd': np.random.uniform(*PID_RANGES['d']),
        }
    
    # Load preprocessed driving data from volume
    # Shape: [n_files, n_steps, 5] - columns: roll_lataccel, v_ego, a_ego, target, steer_command
    print("Loading driving data from volume...")
    driving_data = np.load("/data/driving_data.npy")
    n_total_files, max_steps, _ = driving_data.shape
    print(f"Loaded driving data: {driving_data.shape}")
    
    # Use subset if requested
    if num_files < n_total_files:
        driving_data = driving_data[:num_files]
        print(f"Using first {num_files} files")
    
    n_files = driving_data.shape[0]
    warmup_len = CONTROL_START_IDX - CONTEXT_LENGTH  # 80 steps
    
    # Load transformer model
    print("Loading transformer model...")
    transformer = create_model("/root/models/tinyphysics.onnx")
    
    # Create bicycle model
    print("Creating bicycle model...")
    bicycle = BicycleModel()
    
    def prepare_batch(file_indices):
        """Prepare batch data from driving_data array."""
        batch_data = driving_data[file_indices]  # [batch, steps, 5]
        batch_size = len(file_indices)
        h = min(horizon, max_steps - CONTROL_START_IDX)
        
        # Initial histories (first 20 steps)
        init_actions = jnp.array(batch_data[:, :CONTEXT_LENGTH, 4])  # steer_command
        init_lataccels = jnp.array(batch_data[:, :CONTEXT_LENGTH, 3])  # target
        init_exos = jnp.array(batch_data[:, :CONTEXT_LENGTH, :3])  # roll, v, a
        
        # Warmup phase (steps 20-99)
        warmup_exos = jnp.array(batch_data[:, CONTEXT_LENGTH:CONTROL_START_IDX, :4])  # roll, v, a, target
        warmup_exos = warmup_exos.transpose(1, 0, 2)  # [80, batch, 4]
        warmup_actions = jnp.array(batch_data[:, CONTEXT_LENGTH:CONTROL_START_IDX, 4])  # steer
        warmup_actions = warmup_actions.T  # [80, batch]
        
        # PID phase (steps 100+)
        pid_exos = jnp.array(batch_data[:, CONTROL_START_IDX:CONTROL_START_IDX+h, :4])
        pid_exos = pid_exos.transpose(1, 0, 2)  # [horizon, batch, 4]
        
        return init_actions, init_lataccels, init_exos, warmup_exos, warmup_actions, pid_exos
    
    def run_warmup(init_action_hist, init_lataccel_hist, init_exo_hist, warmup_exos, warmup_actions):
        outputs = run_simulation(
            transformer, init_action_hist, init_lataccel_hist, init_exo_hist,
            warmup_exos, warmup_actions
        )
        final_action_hist = outputs[-CONTEXT_LENGTH:, :, 1].T
        final_lataccel_hist = outputs[-CONTEXT_LENGTH:, :, 0].T
        final_exo_hist = outputs[-CONTEXT_LENGTH:, :, 2:5].transpose(1, 0, 2)
        return final_action_hist, final_lataccel_hist, final_exo_hist
    
    # JIT compile transformer simulations separately (no gradients needed)
    @eqx.filter_jit
    def get_transformer_rollout(init_action_hist, init_lataccel_hist, init_exo_hist,
                                 warmup_exos, warmup_actions, pid_exos, pid_p, pid_i, pid_d):
        """Run transformer (warmup + PID) and return actions + lataccels as fixed data."""
        # Warmup
        post_action, post_lataccel, post_exo = run_warmup(
            init_action_hist, init_lataccel_hist, init_exo_hist,
            warmup_exos, warmup_actions
        )
        
        # PID through transformer
        outputs = run_simulation_pid(
            transformer, post_action, post_lataccel, post_exo, pid_exos,
            p=pid_p, i=pid_i, d=pid_d
        )
        transformer_lataccels = outputs[:, :, 0]  # [horizon, batch]
        pid_actions = outputs[:, :, 1]
        init_lataccel = post_lataccel[:, -1]  # [batch]
        
        return transformer_lataccels, pid_actions, init_lataccel
    
    def loss_fn(bicycle_model, transformer_lataccels, pid_actions, init_lataccel, pid_exos):
        """Compute loss - only bicycle model is differentiable."""
        # Bicycle rollout
        def single_rollout(init_lat, actions, exos):
            return rollout_bicycle(
                bicycle_model, init_lat, actions,
                exos[:, 0], exos[:, 1], exos[:, 2], dt=DT
            )
        
        bicycle_lataccels = jax.vmap(single_rollout)(
            init_lataccel,
            pid_actions.T,
            pid_exos.transpose(1, 0, 2),
        )
        
        mse = jnp.mean((transformer_lataccels.T - bicycle_lataccels) ** 2)
        return mse
    
    @eqx.filter_jit
    def train_step(bicycle_model, opt_state, transformer_lataccels, pid_actions, init_lataccel, pid_exos):
        """Train step - only computes gradients through bicycle model."""
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            bicycle_model, transformer_lataccels, pid_actions, init_lataccel, pid_exos
        )
        
        # Check for NaN
        def check_nan(x):
            if eqx.is_array(x):
                return jnp.any(jnp.isnan(x))
            return False
        
        has_nan = jax.tree_util.tree_reduce(
            lambda a, b: a | b,
            jax.tree_util.tree_map(check_nan, grads),
            initializer=False
        )
        
        def do_update(args):
            grads, opt_state, model = args
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state
        
        def skip_update(args):
            _, opt_state, model = args
            return model, opt_state
        
        bicycle_model, opt_state = jax.lax.cond(
            has_nan, skip_update, do_update,
            (grads, opt_state, bicycle_model)
        )
        
        return bicycle_model, opt_state, loss, has_nan
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(bicycle, eqx.is_array))
    
    n_batches = n_files // batch_size
    
    print(f"\nTraining config:")
    print(f"  Files: {n_files}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Horizon: {horizon}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batches per epoch: {n_batches}")
    
    print(f"\nInitial params:")
    print(f"  steer_ratio={float(bicycle.steer_ratio):.4f}, "
          f"wheelbase={float(bicycle.wheelbase):.4f}, "
          f"roll_coeff={float(bicycle.roll_coeff):.4f}, "
          f"time_const={float(bicycle.time_constant):.4f}")
    
    # Training loop
    for epoch in range(epochs):
        perm = np.random.permutation(n_files)
        epoch_loss = 0.0
        nan_count = 0
        
        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            start = batch_idx * batch_size
            batch_indices = perm[start:start + batch_size]
            
            (init_action, init_lataccel, init_exo, 
             warmup_exos, warmup_actions, pid_exos) = prepare_batch(batch_indices)
            
            pid_gains = sample_pid_gains()
            
            # Run transformer (no gradients needed) - this is the expensive part
            transformer_lataccels, pid_actions, init_lat = get_transformer_rollout(
                init_action, init_lataccel, init_exo,
                warmup_exos, warmup_actions, pid_exos,
                pid_gains['p'], pid_gains['i'], pid_gains['d']
            )
            
            # Train bicycle (cheap - only 4 parameters)
            bicycle, opt_state, loss, had_nan = train_step(
                bicycle, opt_state,
                transformer_lataccels, pid_actions, init_lat, pid_exos
            )
            
            epoch_loss += float(loss)
            nan_count += int(had_nan)
        
        epoch_loss /= n_batches
        nan_str = f" (NaN skipped: {nan_count})" if nan_count > 0 else ""
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}{nan_str}")
        print(f"  steer_ratio={float(bicycle.steer_ratio):.4f}, "
              f"wheelbase={float(bicycle.wheelbase):.4f}, "
              f"roll_coeff={float(bicycle.roll_coeff):.4f}, "
              f"time_const={float(bicycle.time_constant):.4f}")
    
    print("\nTraining complete!")
    
    # Serialize model
    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, bicycle)
    return buffer.getvalue()


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 64,
    horizon: int = 200,
    learning_rate: float = 0.003,
    num_files: int = 10000,
    output_path: str = "bicycle_params_modal.eqx",
):
    """
    Train bicycle model on H100 GPU.
    
    Usage:
        modal run modal_train_bicycle.py --epochs 30 --num-files 10000
    """
    import time
    
    print(f"Training bicycle model on Modal H100")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Horizon: {horizon}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Num files: {num_files}")
    
    start = time.perf_counter()
    
    model_bytes = train_bicycle_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
        horizon=horizon,
        learning_rate=learning_rate,
        num_files=num_files,
    )
    
    elapsed = time.perf_counter() - start
    
    # Save locally
    with open(output_path, "wb") as f:
        f.write(model_bytes)
    
    print(f"\n=== Results ===")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Saved to {output_path} ({len(model_bytes)} bytes)")
