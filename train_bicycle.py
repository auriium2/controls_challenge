"""Train the bicycle model parameters to match the transformer.

The transformer is a black box that maps (state history, action history) -> lataccel.
We want to find bicycle model parameters such that the bicycle model produces
similar lataccels given the same action sequence.

Training procedure:
1. Load CSV data for warmup (first 100 steps have valid steerCommand)
2. Run PID controller through transformer from step 100 onwards
3. This gives us (actions, transformer_lataccels) pairs
4. Run the SAME actions through the bicycle model
5. Compare lataccels and backprop to update bicycle parameters

The transformer is frozen - we don't need STE because we're not backpropping
through the transformer. We just use its outputs as ground truth.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from glob import glob
from tqdm import tqdm
import argparse

from bicycle_model import BicycleModel, BicycleModelExtended, rollout_bicycle
from tinyphysics_eqx import (
    create_model, run_simulation_pid, run_simulation, 
    CONTEXT_LENGTH, MAX_ACC_DELTA
)

ACC_G = 9.81
DT = 0.1  # 10 Hz
CONTROL_START_IDX = 100  # When controller takes over


def load_csv_data(file_path):
    """Load and preprocess CSV data."""
    import pandas as pd
    df = pd.read_csv(file_path)
    return {
        'roll_lataccel': (np.sin(df['roll'].values) * ACC_G).astype(np.float32),
        'v_ego': df['vEgo'].values.astype(np.float32),
        'a_ego': df['aEgo'].values.astype(np.float32),
        'target_lataccel': df['targetLateralAcceleration'].values.astype(np.float32),
        'steer_command': (-df['steerCommand'].values).astype(np.float32),
    }


def prepare_batch_for_pid(file_paths, horizon):
    """Prepare batch data for PID simulation.
    
    Returns initial histories (for warmup) and exo data for the full trajectory.
    The first CONTROL_START_IDX steps use CSV steerCommand (warmup).
    After that, PID controller generates actions.
    
    Args:
        file_paths: list of CSV file paths
        horizon: number of steps after CONTROL_START_IDX to simulate
        
    Returns:
        init_actions: [batch, 20] - action history at start
        init_lataccels: [batch, 20] - lataccel history at start  
        init_exos: [batch, 20, 3] - exo history at start (roll, v, a)
        warmup_exos: [80, batch, 4] - exo data for warmup (steps 20-99)
        warmup_actions: [80, batch] - actions for warmup (steps 20-99)
        pid_exos: [horizon, batch, 4] - exo data for PID phase (steps 100+)
    """
    init_actions = []
    init_lataccels = []
    init_exos = []
    warmup_exos = []
    warmup_actions = []
    pid_exos = []
    
    warmup_len = CONTROL_START_IDX - CONTEXT_LENGTH  # 80 steps
    
    for file_path in file_paths:
        data = load_csv_data(file_path)
        n = len(data['roll_lataccel'])
        h = min(horizon, n - CONTROL_START_IDX)
        
        # Initial histories (first 20 steps)
        init_actions.append(data['steer_command'][:CONTEXT_LENGTH])
        init_lataccels.append(data['target_lataccel'][:CONTEXT_LENGTH])
        init_exos.append(np.stack([
            data['roll_lataccel'][:CONTEXT_LENGTH],
            data['v_ego'][:CONTEXT_LENGTH],
            data['a_ego'][:CONTEXT_LENGTH],
        ], axis=-1))
        
        # Warmup phase (steps 20-99): use CSV steerCommand
        warmup_exo = np.zeros((warmup_len, 4), dtype=np.float32)
        warmup_exo[:, 0] = data['roll_lataccel'][CONTEXT_LENGTH:CONTROL_START_IDX]
        warmup_exo[:, 1] = data['v_ego'][CONTEXT_LENGTH:CONTROL_START_IDX]
        warmup_exo[:, 2] = data['a_ego'][CONTEXT_LENGTH:CONTROL_START_IDX]
        warmup_exo[:, 3] = data['target_lataccel'][CONTEXT_LENGTH:CONTROL_START_IDX]
        warmup_exos.append(warmup_exo)
        
        warmup_act = data['steer_command'][CONTEXT_LENGTH:CONTROL_START_IDX]
        warmup_actions.append(warmup_act)
        
        # PID phase (steps 100+): exo data only, actions come from PID
        pid_exo = np.zeros((horizon, 4), dtype=np.float32)
        pid_exo[:h, 0] = data['roll_lataccel'][CONTROL_START_IDX:CONTROL_START_IDX+h]
        pid_exo[:h, 1] = data['v_ego'][CONTROL_START_IDX:CONTROL_START_IDX+h]
        pid_exo[:h, 2] = data['a_ego'][CONTROL_START_IDX:CONTROL_START_IDX+h]
        pid_exo[:h, 3] = data['target_lataccel'][CONTROL_START_IDX:CONTROL_START_IDX+h]
        pid_exos.append(pid_exo)
    
    return (
        jnp.array(np.stack(init_actions)),       # [batch, 20]
        jnp.array(np.stack(init_lataccels)),     # [batch, 20]
        jnp.array(np.stack(init_exos)),          # [batch, 20, 3]
        jnp.array(np.stack(warmup_exos)).transpose(1, 0, 2),   # [80, batch, 4]
        jnp.array(np.stack(warmup_actions)).T,   # [80, batch]
        jnp.array(np.stack(pid_exos)).transpose(1, 0, 2),      # [horizon, batch, 4]
    )


def run_warmup(model, init_action_hist, init_lataccel_hist, init_exo_hist, 
               warmup_exos, warmup_actions):
    """Run warmup phase using CSV steerCommand values.
    
    Returns updated histories after warmup, ready for PID phase.
    """
    outputs = run_simulation(
        model, init_action_hist, init_lataccel_hist, init_exo_hist,
        warmup_exos, warmup_actions
    )
    # outputs: [80, batch, 6] = (lataccel, action, roll, v, a, target)
    
    # Extract final histories
    batch_size = init_action_hist.shape[0]
    final_action_hist = outputs[-CONTEXT_LENGTH:, :, 1].T  # [batch, 20]
    final_lataccel_hist = outputs[-CONTEXT_LENGTH:, :, 0].T  # [batch, 20]
    final_exo_hist = outputs[-CONTEXT_LENGTH:, :, 2:5].transpose(1, 0, 2)  # [batch, 20, 3]
    
    return final_action_hist, final_lataccel_hist, final_exo_hist


def make_pid_step(p=0.195, i=0.100, d=-0.053):
    """Create PID step function for use with lax.scan."""
    def pid_step(carry, inputs):
        error_integral, prev_error = carry
        current_lataccel, target = inputs
        
        error = target - current_lataccel
        error_integral_new = error_integral + error
        error_diff = error - prev_error
        
        action = p * error + i * error_integral_new + d * error_diff
        action = jnp.clip(action, -2, 2)
        
        return (error_integral_new, error), action
    return pid_step


def run_pid_simulation_batched(model, action_hist, lataccel_hist, exo_hist, 
                                pid_exos, p=0.195, i=0.100, d=-0.053):
    """Run PID controller through transformer, return actions and lataccels.
    
    This runs the HARD simulation (no STE needed since we don't backprop through it).
    
    Returns:
        actions: [horizon, batch] - PID-generated actions
        lataccels: [horizon, batch] - transformer outputs
    """
    outputs = run_simulation_pid(
        model, action_hist, lataccel_hist, exo_hist, pid_exos,
        p=p, i=i, d=d
    )
    # outputs: [horizon, batch, 6] = (lataccel, action, roll, v, a, target)
    
    lataccels = outputs[:, :, 0]  # [horizon, batch]
    actions = outputs[:, :, 1]    # [horizon, batch]
    
    return actions, lataccels


# Baseline PID gains and safe variation ranges
BASELINE_PID = {'p': 0.195, 'i': 0.100, 'd': -0.053}
# Stay within ~50% of baseline to avoid saturation
PID_RANGES = {
    'p': (0.10, 0.30),   # baseline 0.195
    'i': (0.05, 0.15),   # baseline 0.100
    'd': (-0.10, -0.02), # baseline -0.053
}


def sample_pid_gains():
    """Sample random PID gains from safe ranges."""
    return {
        'p': np.random.uniform(*PID_RANGES['p']),
        'i': np.random.uniform(*PID_RANGES['i']),
        'd': np.random.uniform(*PID_RANGES['d']),
    }


def make_loss_fn(transformer_model):
    """Create loss function comparing bicycle to transformer rollouts.
    
    The transformer is frozen - we just use its outputs as ground truth.
    """
    
    def loss_fn(bicycle_model, 
                init_action_hist, init_lataccel_hist, init_exo_hist,
                warmup_exos, warmup_actions, pid_exos,
                pid_p, pid_i, pid_d):
        """
        Full training loss:
        1. Run warmup through transformer (using CSV actions)
        2. Run PID through transformer (generates actions + lataccels)
        3. Run bicycle model with same actions
        4. Compare lataccels
        """
        batch_size = init_action_hist.shape[0]
        horizon = pid_exos.shape[0]
        
        # Step 1: Warmup phase - run through transformer with CSV actions
        post_warmup_action_hist, post_warmup_lataccel_hist, post_warmup_exo_hist = run_warmup(
            transformer_model,
            init_action_hist, init_lataccel_hist, init_exo_hist,
            warmup_exos, warmup_actions
        )
        
        # Step 2: PID phase through transformer (with varied gains)
        pid_actions, transformer_lataccels = run_pid_simulation_batched(
            transformer_model,
            post_warmup_action_hist, post_warmup_lataccel_hist, post_warmup_exo_hist,
            pid_exos,
            p=pid_p, i=pid_i, d=pid_d
        )
        # pid_actions: [horizon, batch], transformer_lataccels: [horizon, batch]
        
        # Step 3: Run bicycle model with the SAME actions
        def single_bicycle_rollout(init_lat, actions, exos):
            # actions: [horizon], exos: [horizon, 4] = (roll, v, a, target)
            return rollout_bicycle(
                bicycle_model,
                init_lat,
                actions,
                exos[:, 0],  # roll
                exos[:, 1],  # v
                exos[:, 2],  # a
                dt=DT,
            )
        
        bicycle_lataccels = jax.vmap(single_bicycle_rollout)(
            post_warmup_lataccel_hist[:, -1],  # [batch] - init lataccel
            pid_actions.T,   # [batch, horizon]
            pid_exos.transpose(1, 0, 2),  # [batch, horizon, 4]
        )  # [batch, horizon]
        
        # Step 4: Compare
        # transformer_lataccels: [horizon, batch] -> [batch, horizon]
        mse = jnp.mean((transformer_lataccels.T - bicycle_lataccels) ** 2)
        
        return mse
    
    return loss_fn


@eqx.filter_jit
def train_step(bicycle_model, opt_state, optimizer, loss_fn,
               init_action_hist, init_lataccel_hist, init_exo_hist,
               warmup_exos, warmup_actions, pid_exos,
               pid_p, pid_i, pid_d):
    """Single training step."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        bicycle_model,
        init_action_hist, init_lataccel_hist, init_exo_hist,
        warmup_exos, warmup_actions, pid_exos,
        pid_p, pid_i, pid_d
    )
    
    # Check for NaN gradients
    def check_nan(x):
        if eqx.is_array(x):
            return jnp.any(jnp.isnan(x))
        return False
    
    has_nan = jax.tree_util.tree_reduce(
        lambda a, b: a | b,
        jax.tree_util.tree_map(check_nan, grads),
        initializer=False
    )
    
    # Only update if no NaN
    def do_update(args):
        grads, opt_state, bicycle_model = args
        updates, new_opt_state = optimizer.update(grads, opt_state, bicycle_model)
        new_model = eqx.apply_updates(bicycle_model, updates)
        return new_model, new_opt_state
    
    def skip_update(args):
        _, opt_state, bicycle_model = args
        return bicycle_model, opt_state
    
    bicycle_model, opt_state = jax.lax.cond(
        has_nan,
        skip_update,
        do_update,
        (grads, opt_state, bicycle_model)
    )
    
    return bicycle_model, opt_state, loss, has_nan


def print_params(model):
    """Print current bicycle model parameters."""
    if isinstance(model, BicycleModelExtended):
        print(f"  steer_ratio={float(model.steer_ratio):.4f}, "
              f"wheelbase={float(model.wheelbase):.4f}, "
              f"roll_coeff={float(model.roll_coeff):.4f}, "
              f"time_const={float(model.time_constant):.4f}, "
              f"v_steer={float(model.v_steer_coeff):.6f}, "
              f"accel={float(model.accel_coeff):.4f}, "
              f"bias={float(model.bias):.4f}")
    else:
        print(f"  steer_ratio={float(model.steer_ratio):.4f}, "
              f"wheelbase={float(model.wheelbase):.4f}, "
              f"roll_coeff={float(model.roll_coeff):.4f}, "
              f"time_const={float(model.time_constant):.4f}")


def train(
    bicycle_model,
    transformer_model,
    train_files,
    num_epochs=50,
    batch_size=16,
    horizon=100,
    learning_rate=0.01,
):
    """Train bicycle model parameters."""
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(eqx.filter(bicycle_model, eqx.is_array))
    
    loss_fn = make_loss_fn(transformer_model)
    
    n_batches = len(train_files) // batch_size
    
    print("Initial parameters:")
    print_params(bicycle_model)
    
    for epoch in range(num_epochs):
        perm = np.random.permutation(len(train_files))
        shuffled_files = [train_files[i] for i in perm]
        
        epoch_loss = 0.0
        nan_count = 0
        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            start = batch_idx * batch_size
            batch_files = shuffled_files[start:start + batch_size]
            
            (init_action, init_lataccel, init_exo, 
             warmup_exos, warmup_actions, pid_exos) = prepare_batch_for_pid(batch_files, horizon)
            
            # Sample random PID gains for this batch
            pid_gains = sample_pid_gains()
            
            bicycle_model, opt_state, loss, had_nan = train_step(
                bicycle_model, opt_state, optimizer, loss_fn,
                init_action, init_lataccel, init_exo,
                warmup_exos, warmup_actions, pid_exos,
                pid_gains['p'], pid_gains['i'], pid_gains['d']
            )
            
            epoch_loss += float(loss)
            nan_count += int(had_nan)
        
        epoch_loss /= n_batches
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}" + 
              (f" (NaN skipped: {nan_count})" if nan_count > 0 else ""))
        print_params(bicycle_model)
    
    return bicycle_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-files', type=int, default=1000)
    parser.add_argument('--extended', action='store_true', help='Use extended model')
    parser.add_argument('--output', type=str, default='bicycle_params.eqx')
    args = parser.parse_args()
    
    print("Loading transformer model...")
    transformer = create_model('models/tinyphysics.onnx')
    
    if args.extended:
        print("Creating extended bicycle model")
        bicycle = BicycleModelExtended()
    else:
        print("Creating basic bicycle model")
        bicycle = BicycleModel()
    
    train_files = sorted(glob('data/*.csv'))[:args.num_files]
    print(f"Training on {len(train_files)} files")
    print(f"  warmup: {CONTROL_START_IDX - CONTEXT_LENGTH} steps (20-99), using CSV actions")
    print(f"  PID phase: {args.horizon} steps (100+), using PID-generated actions")
    
    bicycle = train(
        bicycle,
        transformer,
        train_files,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        horizon=args.horizon,
        learning_rate=args.lr,
    )
    
    print("\nFinal parameters:")
    print_params(bicycle)
    
    eqx.tree_serialise_leaves(args.output, bicycle)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
