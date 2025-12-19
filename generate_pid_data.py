"""Generate training data by running PID simulations through the transformer model.

This script runs the tinyphysics_eqx model with a noisy PID controller to collect
(state, action) -> next_lataccel transition data for training a bicycle model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
from functools import partial
from glob import glob
from tqdm import tqdm
from collections import defaultdict

from tinyphysics_eqx import (
    create_model,
    run_simulation_pid,
    run_simulation_noisy_pid,
    CONTEXT_LENGTH,
)

ACC_G = 9.81
MIN_SEQUENCE_LENGTH = 200  # Minimum steps after context to be useful


def load_csv_data(file_path):
    """Load data from a CSV file with proper preprocessing."""
    import pandas as pd
    df = pd.read_csv(file_path)
    return {
        # Match tinyphysics.py preprocessing
        'roll_lataccel': (np.sin(df['roll'].values) * ACC_G).astype(np.float32),
        'v_ego': df['vEgo'].values.astype(np.float32),
        'a_ego': df['aEgo'].values.astype(np.float32),
        'target_lataccel': df['targetLateralAcceleration'].values.astype(np.float32),
        'steer_command': (-df['steerCommand'].values).astype(np.float32),  # Negate like tinyphysics
    }


def get_file_length(file_path):
    """Get the number of rows in a CSV file quickly."""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f) - 1  # Subtract header


def make_batched_simulation(model, noise_std):
    """Create JIT-compiled batched noisy PID simulation using vmap."""
    
    def run_single(init_action, init_lataccel, init_exo, exo_data, key):
        """Run single noisy PID simulation.
        
        Args:
            init_action: [20] 
            init_lataccel: [20]
            init_exo: [20, 3]
            exo_data: [n_steps, 4]
            key: PRNGKey
        """
        # Add batch dimension of 1
        return run_simulation_noisy_pid(
            model, 
            init_action[None, :], 
            init_lataccel[None, :], 
            init_exo[None, :, :], 
            exo_data[:, None, :],  # [n_steps, 1, 4]
            key, 
            noise_std=noise_std
        )[:, 0, :]  # Remove batch dim: [n_steps, 6]
    
    # vmap over batch dimension (axis 0 for all inputs)
    run_batched = jax.vmap(run_single, in_axes=(0, 0, 0, 1, 0))
    
    @jax.jit
    def run_batch(init_action, init_lataccel, init_exo, exo_data, keys):
        """Run batched noisy PID simulation.
        
        Args:
            init_action: [batch, 20]
            init_lataccel: [batch, 20]
            init_exo: [batch, 20, 3]
            exo_data: [n_steps, batch, 4]
            keys: [batch, 2] PRNGKeys
            
        Returns:
            outputs: [batch, n_steps, 6]
        """
        return run_batched(init_action, init_lataccel, init_exo, exo_data, keys)
    
    return run_batch


def prepare_batch_data(file_paths, target_length):
    """Prepare batched data from files, truncating to target_length.
    
    Returns:
        Batched arrays ready for simulation
    """
    batch_size = len(file_paths)
    
    init_actions = np.zeros((batch_size, CONTEXT_LENGTH), dtype=np.float32)
    init_lataccels = np.zeros((batch_size, CONTEXT_LENGTH), dtype=np.float32)
    init_exos = np.zeros((batch_size, CONTEXT_LENGTH, 3), dtype=np.float32)
    exo_datas = np.zeros((target_length, batch_size, 4), dtype=np.float32)
    
    valid_lengths = []
    
    for i, file_path in enumerate(file_paths):
        data = load_csv_data(file_path)
        n_total = len(data['roll_lataccel'])
        n_steps = min(n_total - CONTEXT_LENGTH, target_length)
        valid_lengths.append(n_steps)
        
        init_actions[i] = data['steer_command'][:CONTEXT_LENGTH]
        init_lataccels[i] = data['target_lataccel'][:CONTEXT_LENGTH]
        init_exos[i, :, 0] = data['roll_lataccel'][:CONTEXT_LENGTH]
        init_exos[i, :, 1] = data['v_ego'][:CONTEXT_LENGTH]
        init_exos[i, :, 2] = data['a_ego'][:CONTEXT_LENGTH]
        
        exo_datas[:n_steps, i, 0] = data['roll_lataccel'][CONTEXT_LENGTH:CONTEXT_LENGTH + n_steps]
        exo_datas[:n_steps, i, 1] = data['v_ego'][CONTEXT_LENGTH:CONTEXT_LENGTH + n_steps]
        exo_datas[:n_steps, i, 2] = data['a_ego'][CONTEXT_LENGTH:CONTEXT_LENGTH + n_steps]
        exo_datas[:n_steps, i, 3] = data['target_lataccel'][CONTEXT_LENGTH:CONTEXT_LENGTH + n_steps]
    
    return (
        jnp.array(init_actions),
        jnp.array(init_lataccels),
        jnp.array(init_exos),
        jnp.array(exo_datas),
        valid_lengths
    )


def extract_transitions(outputs, valid_lengths):
    """Extract (state, action) -> next_lataccel transitions from simulation outputs.
    
    Args:
        outputs: [batch, n_steps, 6] - (lataccel, action, roll, v, a, target)
        valid_lengths: List of valid lengths per batch element
        
    Returns:
        states: [total_transitions, 5]
        next_lataccels: [total_transitions]
    """
    outputs = np.array(outputs)
    all_states = []
    all_next_lataccels = []
    
    for b, valid_len in enumerate(valid_lengths):
        if valid_len < 2:
            continue
            
        lataccels = outputs[b, :valid_len, 0]
        actions = outputs[b, :valid_len, 1]
        rolls = outputs[b, :valid_len, 2]
        vs = outputs[b, :valid_len, 3]
        a_s = outputs[b, :valid_len, 4]
        
        # State at t, next_lataccel at t+1
        states = np.stack([
            lataccels[:-1], 
            actions[:-1], 
            rolls[:-1], 
            vs[:-1], 
            a_s[:-1]
        ], axis=-1)
        next_lataccels = lataccels[1:]
        
        all_states.append(states)
        all_next_lataccels.append(next_lataccels)
    
    if not all_states:
        return np.array([]).reshape(0, 5), np.array([])
    
    return np.concatenate(all_states), np.concatenate(all_next_lataccels)


def generate_training_data(
    model,
    data_files,
    output_path,
    noise_std=0.3,
    seed=42,
    batch_size=32,
    sequence_length=500,
):
    """Generate training data from multiple files with batched JIT simulation.
    
    Args:
        model: TinyPhysicsModel
        data_files: List of CSV file paths
        output_path: Path to save the .npz file
        noise_std: Standard deviation of PID noise for exploration
        seed: Random seed
        batch_size: Number of files to process in parallel
        sequence_length: Truncate sequences to this length
    """
    key = jax.random.PRNGKey(seed)
    
    # Filter files by minimum length
    print("Filtering files by length...")
    valid_files = []
    for f in tqdm(data_files, desc="Checking file lengths"):
        length = get_file_length(f)
        if length >= CONTEXT_LENGTH + MIN_SEQUENCE_LENGTH:
            valid_files.append(f)
    
    print(f"Kept {len(valid_files)}/{len(data_files)} files with >= {MIN_SEQUENCE_LENGTH} steps")
    
    # Create JIT-compiled simulation
    run_batch = make_batched_simulation(model, noise_std)
    
    all_states = []
    all_next_lataccels = []
    
    # Process in batches
    n_batches = (len(valid_files) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(valid_files))
        batch_files = valid_files[start:end]
        
        if not batch_files:
            continue
        
        # Prepare batch data
        init_action, init_lataccel, init_exo, exo_data, valid_lengths = prepare_batch_data(
            batch_files, sequence_length
        )
        
        # Generate random keys for each batch element
        key, *subkeys = jax.random.split(key, len(batch_files) + 1)
        keys = jnp.stack(subkeys)
        
        # Run simulation
        outputs = run_batch(
            init_action, init_lataccel, init_exo, exo_data, keys
        )
        
        # Extract transitions
        states, next_lataccels = extract_transitions(outputs, valid_lengths)
        
        if len(states) > 0:
            all_states.append(states)
            all_next_lataccels.append(next_lataccels)
    
    # Concatenate all data
    states = np.concatenate(all_states, axis=0)
    next_lataccels = np.concatenate(all_next_lataccels, axis=0)
    
    print(f"\nCollected {len(states):,} transitions")
    print(f"States shape: {states.shape}")
    print(f"Next lataccels shape: {next_lataccels.shape}")
    
    # Print statistics
    print(f"\nState statistics:")
    print(f"  lataccel: mean={states[:, 0].mean():.3f}, std={states[:, 0].std():.3f}")
    print(f"  action:   mean={states[:, 1].mean():.3f}, std={states[:, 1].std():.3f}")
    print(f"  roll:     mean={states[:, 2].mean():.3f}, std={states[:, 2].std():.3f}")
    print(f"  v_ego:    mean={states[:, 3].mean():.3f}, std={states[:, 3].std():.3f}")
    print(f"  a_ego:    mean={states[:, 4].mean():.3f}, std={states[:, 4].std():.3f}")
    print(f"  next_lat: mean={next_lataccels.mean():.3f}, std={next_lataccels.std():.3f}")
    
    # Save
    np.savez(
        output_path,
        states=states.astype(np.float32),
        next_lataccels=next_lataccels.astype(np.float32),
    )
    print(f"\nSaved to {output_path}")
    
    return states, next_lataccels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-files', type=int, default=1000, help='Number of files to process')
    parser.add_argument('--noise-std', type=float, default=0.3, help='PID noise std')
    parser.add_argument('--output', type=str, default='bicycle_training_data.npz', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for parallel processing')
    parser.add_argument('--sequence-length', type=int, default=500, help='Truncate sequences to this length')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = create_model('models/tinyphysics.onnx')
    
    # Get data files
    data_files = sorted(glob('data/*.csv'))[:args.num_files]
    print(f"Found {len(data_files)} files")
    
    # Generate data
    generate_training_data(
        model,
        data_files,
        args.output,
        noise_std=args.noise_std,
        seed=args.seed,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )


if __name__ == '__main__':
    main()
