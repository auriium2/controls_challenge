"""Train all bicycle model variants on Modal H100."""

import modal

app = modal.App("bicycle-model-training")

volume = modal.Volume.from_name("koopman-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]==0.6.0",
        "equinox>=0.11.0",
        "numpy>=1.24.0",
        "optax>=0.1.7",
        "tqdm>=4.66.0",
    )
    .add_local_file("bicycle_model.py", remote_path="/root/bicycle_model.py")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/data": volume},
)
def train_model(model_type: str, data_path: str, epochs: int, batch_size: int, lr: float) -> dict:
    """Train a single bicycle model variant."""
    import sys
    sys.path.insert(0, "/root")
    
    import io
    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    from tqdm import tqdm
    
    from bicycle_model import (
        BicycleModel, BicycleModelExtended,
        BicycleModelTan, BicycleModelTanExtended,
        BicycleModelLinear, BicycleModelQuadratic,
        rollout_bicycle
    )
    
    MODEL_TYPES = {
        'basic': BicycleModel,
        'extended': BicycleModelExtended,
        'tan': BicycleModelTan,
        'tan_extended': BicycleModelTanExtended,
        'linear': BicycleModelLinear,
        'quadratic': BicycleModelQuadratic,
    }
    
    DT = 0.1
    
    print(f"JAX devices: {jax.devices()}")
    print(f"Training model: {model_type}")
    
    # Load data from volume
    data = np.load(data_path)
    trajectories = data['trajectories']
    init_lataccels = data['init_lataccels']
    
    n_traj, horizon, _ = trajectories.shape
    print(f"Loaded {n_traj} trajectories, horizon={horizon}")
    
    # Extract columns
    transformer_lataccels = jnp.array(trajectories[:, :, 0])
    actions = jnp.array(trajectories[:, :, 1])
    exo_data = jnp.array(trajectories[:, :, 2:6])
    init_lats = jnp.array(init_lataccels)
    
    # Create model
    bicycle = MODEL_TYPES[model_type]()
    
    # Loss function
    def loss_fn(model, trans_lat, acts, exos, init_lat):
        def single_rollout(init_l, a, e):
            return rollout_bicycle(model, init_l, a, e[:, 0], e[:, 1], e[:, 2], dt=DT)
        
        pred = jax.vmap(single_rollout)(init_lat, acts, exos)
        return jnp.mean((trans_lat - pred) ** 2)
    
    @eqx.filter_jit
    def train_step(model, opt_state, trans_lat, acts, exos, init_lat):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, trans_lat, acts, exos, init_lat)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    # Train
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(bicycle, eqx.is_array))
    
    n_batches = n_traj // batch_size
    losses = []
    
    for epoch in range(epochs):
        perm = np.random.permutation(n_traj)
        epoch_loss = 0.0
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            idx = perm[start:start + batch_size]
            
            bicycle, opt_state, loss = train_step(
                bicycle, opt_state,
                transformer_lataccels[idx],
                actions[idx],
                exo_data[idx],
                init_lats[idx],
            )
            epoch_loss += float(loss)
        
        epoch_loss /= n_batches
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.6f}")
    
    # Extract final parameters
    params = {}
    for name in dir(bicycle):
        if not name.startswith('_'):
            val = getattr(bicycle, name)
            if hasattr(val, 'item'):
                params[name] = float(val)
    
    return {
        'model_type': model_type,
        'final_loss': losses[-1],
        'losses': losses,
        'params': params,
    }


@app.function(volumes={"/data": volume})
def upload_data(data_bytes: bytes, remote_name: str):
    """Upload data file to volume."""
    with open(f"/data/{remote_name}", "wb") as f:
        f.write(data_bytes)
    volume.commit()
    print(f"Uploaded {len(data_bytes)/1024/1024:.1f} MB to /data/{remote_name}")


@app.local_entrypoint()
def main(
    data_path: str = "bicycle_training_diverse_clean.npz",
    epochs: int = 20,
    batch_size: int = 2000,
    lr: float = 0.01,
    upload: bool = False,
):
    """Train all model variants."""
    import time
    
    remote_data_path = "/data/bicycle_training_diverse_clean.npz"
    
    if upload:
        print(f"Uploading {data_path} to volume...")
        with open(data_path, "rb") as f:
            data_bytes = f.read()
        upload_data.remote(data_bytes, "bicycle_training_diverse_clean.npz")
        print("Upload complete!")
    
    model_types = ['basic', 'extended', 'tan', 'tan_extended', 'linear', 'quadratic']
    
    print(f"\nTraining {len(model_types)} model variants...")
    print(f"Epochs: {epochs}, batch_size: {batch_size}, lr: {lr}")
    print("=" * 60)
    
    # Launch all in parallel
    start = time.perf_counter()
    futures = []
    for model_type in model_types:
        future = train_model.spawn(model_type, remote_data_path, epochs, batch_size, lr)
        futures.append((model_type, future))
    
    # Collect results
    results = []
    for model_type, future in futures:
        result = future.get()
        results.append(result)
        print(f"\n{model_type}: loss={result['final_loss']:.6f}")
        print(f"  params: {result['params']}")
    
    elapsed = time.perf_counter() - start
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results.sort(key=lambda x: x['final_loss'])
    for r in results:
        print(f"{r['model_type']:15s}: loss={r['final_loss']:.6f}")
    
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Best model: {results[0]['model_type']} (loss={results[0]['final_loss']:.6f})")
