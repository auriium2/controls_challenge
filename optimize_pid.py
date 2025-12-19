"""Optimize PID gains via gradient descent through the differentiable transformer."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from tinyphysics_eqx import (create_model, run_simulation_ste_pid, run_simulation_soft_pid, 
                             run_simulation_ce_pid, run_simulation_sparsemax_pid, CONTEXT_LENGTH, BINS)

# Cost function constants from tinyphysics.py
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
CONTROL_START_IDX = 100
WARMUP_HORIZON = 80  # Steps 20-100 (CONTEXT_LENGTH to CONTROL_START_IDX) 
CONTROL_HORIZON = 200  # Steps 100-300 (where we actually control) - shorter for faster training


def compute_cost(pred_lataccel, target_lataccel):
    """Compute cost matching eval.py exactly."""
    lataccel_cost = jnp.mean((target_lataccel - pred_lataccel) ** 2) * 100
    jerk_cost = jnp.mean((jnp.diff(pred_lataccel) / DEL_T) ** 2) * 100
    total_cost = lataccel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
    return total_cost, lataccel_cost, jerk_cost


def load_batched_data(data_dir, n_files):
    """Load and batch driving data with warmup and control periods separated.
    
    IMPORTANT: During warmup (steps 0-100), tinyphysics.py uses target_lataccel 
    as current_lataccel (ground truth). The model prediction is ignored.
    Also computes accumulated PID state (error_integral, prev_error) from warmup.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.iterdir())[:n_files]
    
    min_len = CONTROL_START_IDX + CONTROL_HORIZON
    
    all_post_warmup_action = []
    all_post_warmup_lataccel = []
    all_post_warmup_exo = []
    all_control_exo = []
    all_error_integral = []
    all_prev_error = []
    
    for f in tqdm(files, desc="Loading data"):
        df = pd.read_csv(f)
        if len(df) < min_len:
            continue
            
        ACC_G = 9.81
        roll_lataccel = np.sin(df['roll'].values) * ACC_G  # Convert roll to lataccel
        v_ego = df['vEgo'].values
        a_ego = df['aEgo'].values
        target = df['targetLateralAcceleration'].values
        steer = -df['steerCommand'].values  # NEGATE per tinyphysics.py
        
        exo = np.stack([roll_lataccel, v_ego, a_ego, target], axis=-1)
        
        # Compute accumulated PID state from warmup
        # During warmup, current_lataccel[t] = target[t-1] (1-step lag)
        # At step 20 (CONTEXT_LENGTH), current = target[19]
        # error[t] = target[t] - current[t] = target[t] - target[t-1]
        error_integral = 0.0
        prev_error = 0.0
        for t in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            current = target[t-1]  # current_lataccel is previous target
            error = target[t] - current
            error_integral += error
            prev_error = error
        
        all_error_integral.append(error_integral)
        all_prev_error.append(prev_error)
        
        # At step 100, we have last 20 steps of histories
        warmup_end = CONTROL_START_IDX
        all_post_warmup_action.append(steer[warmup_end-CONTEXT_LENGTH:warmup_end])
        all_post_warmup_lataccel.append(target[warmup_end-CONTEXT_LENGTH:warmup_end])
        all_post_warmup_exo.append(exo[warmup_end-CONTEXT_LENGTH:warmup_end, :3])
        
        # Control period (steps 100 to 100+CONTROL_HORIZON)
        all_control_exo.append(exo[CONTROL_START_IDX:CONTROL_START_IDX+CONTROL_HORIZON])
    
    return {
        'post_warmup_action_hist': jnp.array(all_post_warmup_action),      # [batch, 20]
        'post_warmup_lataccel_hist': jnp.array(all_post_warmup_lataccel),  # [batch, 20]
        'post_warmup_exo_hist': jnp.array(all_post_warmup_exo),            # [batch, 20, 3]
        'control_exo': jnp.array(all_control_exo).transpose(1, 0, 2),      # [horizon, batch, 4]
        'error_integral': jnp.array(all_error_integral),                    # [batch]
        'prev_error': jnp.array(all_prev_error),                            # [batch]
        'n_files': len(all_post_warmup_action),
    }


def make_pid_rollout_fn(model, mode='sparsemax', temperature=0.5):
    """Create PID rollout function. Warmup state comes from data (ground truth).
    
    Args:
        model: TinyPhysicsModel
        mode: 'sparsemax' (sparse but differentiable), 'ce' (soft), 'hard' (STE)
        temperature: Softmax temperature (only for ce mode).
    """
    
    def pid_rollout(pid_params, post_warmup_action_hist, post_warmup_lataccel_hist, 
                    post_warmup_exo_hist, control_exo, error_integral, prev_error):
        p, i, d = pid_params[0], pid_params[1], pid_params[2]
        
        if mode == 'sparsemax':
            return run_simulation_sparsemax_pid(
                model,
                post_warmup_action_hist,
                post_warmup_lataccel_hist,
                post_warmup_exo_hist,
                control_exo,
                p, i, d,
                init_error_integral=error_integral,
                init_prev_error=prev_error,
            )
        elif mode == 'ce':
            return run_simulation_ce_pid(
                model,
                post_warmup_action_hist,
                post_warmup_lataccel_hist,
                post_warmup_exo_hist,
                control_exo,
                p, i, d,
                init_error_integral=error_integral,
                init_prev_error=prev_error,
                temperature=temperature
            )
        else:  # hard
            return run_simulation_ste_pid(
                model,
                post_warmup_action_hist,
                post_warmup_lataccel_hist,
                post_warmup_exo_hist,
                control_exo,
                p, i, d,
                init_error_integral=error_integral,
                init_prev_error=prev_error,
                temperature=0.1
            )
    
    return pid_rollout


def get_target_tokens(target_values):
    """Get token indices for target values."""
    return jnp.argmin(jnp.abs(BINS[None, :] - target_values[:, None]), axis=1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-files', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.1)  # Lower temp = closer to hard
    parser.add_argument('--mode', type=str, default='sparsemax', choices=['sparsemax', 'soft_mse', 'ce'])
    args = parser.parse_args()
    
    print("Loading transformer model...")
    model = create_model("models/tinyphysics.onnx")
    
    print(f"Loading {args.n_files} driving data files...")
    data = load_batched_data(args.data_dir, args.n_files)
    print(f"Loaded {data['n_files']} files")
    print(f"Warmup horizon: {WARMUP_HORIZON}, Control horizon: {CONTROL_HORIZON}")
    
    # Create rollout functions
    rollout_fn_sparsemax = make_pid_rollout_fn(model, mode='sparsemax')
    rollout_fn_soft = make_pid_rollout_fn(model, mode='ce', temperature=args.temperature)  # returns logits
    rollout_fn_hard = make_pid_rollout_fn(model, mode='hard')
    
    # Sparsemax loss - sparse but differentiable, outputs often match hard exactly
    def loss_fn_sparsemax(pid_params, data):
        outputs = rollout_fn_sparsemax(
            pid_params,
            data['post_warmup_action_hist'],
            data['post_warmup_lataccel_hist'],
            data['post_warmup_exo_hist'],
            data['control_exo'],
            data['error_integral'],
            data['prev_error'],
        )
        # outputs: [horizon, batch, 3] = (lataccel, action, target)
        lataccels = outputs[:, :, 0]
        targets = outputs[:, :, 2]
        
        # MSE tracking error - SAME as eval metric
        lataccel_cost = jnp.mean((targets - lataccels) ** 2) * 100
        
        # Jerk cost
        jerk = jnp.diff(lataccels, axis=0) / DEL_T
        jerk_cost = jnp.mean(jerk ** 2) * 100
        
        total_cost = lataccel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
        return total_cost
    
    # Soft MSE loss - directly optimizes tracking error on soft simulation
    def loss_fn_soft_mse(pid_params, data):
        logits, lataccels, actions, targets = rollout_fn_soft(
            pid_params,
            data['post_warmup_action_hist'],
            data['post_warmup_lataccel_hist'],
            data['post_warmup_exo_hist'],
            data['control_exo'],
            data['error_integral'],
            data['prev_error'],
        )
        # lataccels: [horizon, batch], targets: [horizon, batch]
        
        # MSE tracking error - SAME as eval metric
        lataccel_cost = jnp.mean((targets - lataccels) ** 2) * 100
        
        # Jerk cost
        jerk = jnp.diff(lataccels, axis=0) / DEL_T
        jerk_cost = jnp.mean(jerk ** 2) * 100
        
        total_cost = lataccel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
        return total_cost
    
    # Cross-entropy loss function
    def loss_fn_ce(pid_params, data):
        logits, lataccels, actions, targets = rollout_fn_soft(
            pid_params,
            data['post_warmup_action_hist'],
            data['post_warmup_lataccel_hist'],
            data['post_warmup_exo_hist'],
            data['control_exo'],
            data['error_integral'],
            data['prev_error'],
        )
        horizon, batch, vocab = logits.shape
        logits_flat = logits.reshape(-1, vocab)
        targets_flat = targets.reshape(-1)
        target_tokens = jnp.argmin(jnp.abs(BINS[None, :] - targets_flat[:, None]), axis=1)
        log_probs = jax.nn.log_softmax(logits_flat)
        ce_loss = -jnp.mean(log_probs[jnp.arange(len(target_tokens)), target_tokens])
        jerk = jnp.diff(lataccels, axis=0) / DEL_T
        jerk_loss = jnp.mean(jerk ** 2)
        total_loss = ce_loss * 100 + jerk_loss * 100
        return total_loss
    
    # Eval function (uses hard simulation - matches real ONNX behavior)
    def loss_fn_hard(pid_params, data):
        outputs = rollout_fn_hard(
            pid_params,
            data['post_warmup_action_hist'],
            data['post_warmup_lataccel_hist'],
            data['post_warmup_exo_hist'],
            data['control_exo'],
            data['error_integral'],
            data['prev_error'],
        )
        pred_lataccel = outputs[:, :, 0]
        target_lataccel = outputs[:, :, 2]
        total_cost, _, _ = compute_cost(pred_lataccel.T.flatten(), target_lataccel.T.flatten())
        return total_cost
    
    # Select loss function
    if args.mode == 'sparsemax':
        loss_fn = loss_fn_sparsemax
        print(f"\nUsing SPARSEMAX loss (sparse but differentiable)")
    elif args.mode == 'soft_mse':
        loss_fn = loss_fn_soft_mse
        print(f"\nUsing SOFT MSE loss (temp={args.temperature})")
    else:
        loss_fn = loss_fn_ce
        print(f"\nUsing CE loss (temp={args.temperature})")
    
    # JIT compile
    loss_and_grad = eqx.filter_jit(jax.value_and_grad(loss_fn))
    eval_loss = eqx.filter_jit(loss_fn_hard)
    
    # Initial PID params (baseline from controllers/pid.py)
    pid_params = jnp.array([0.195, 0.1, -0.053])
    
    print(f"Baseline PID: p=0.195, i=0.1, d=-0.053")
    
    # Compute baseline costs
    print("Computing initial costs (compiling)...")
    init_loss, init_grad = loss_and_grad(pid_params, data)
    init_hard = eval_loss(pid_params, data)
    print(f"Initial soft cost: {float(init_loss):.2f}")
    print(f"Initial hard cost: {float(init_hard):.2f}")
    print(f"Initial grad: {init_grad}")
    
    # Optimizer  
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(pid_params)
    
    print(f"\nOptimizing over {args.epochs} epochs...")
    
    best_hard_cost = float('inf')
    best_params = pid_params
    
    for epoch in range(args.epochs):
        loss, grad = loss_and_grad(pid_params, data)
        
        updates, opt_state = optimizer.update(grad, opt_state, pid_params)
        pid_params = optax.apply_updates(pid_params, updates)
        
        # Clip to reasonable ranges
        pid_params = jnp.clip(pid_params, jnp.array([0.01, 0.0, -1.0]), jnp.array([1.0, 0.5, 0.0]))
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            hard_cost = eval_loss(pid_params, data)
            if hard_cost < best_hard_cost:
                best_hard_cost = hard_cost
                best_params = pid_params
            p, i, d = pid_params
            print(f"Epoch {epoch+1}: soft={float(loss):.2f}, hard={float(hard_cost):.2f}, p={float(p):.4f}, i={float(i):.4f}, d={float(d):.4f}")
    
    # Final comparison using HARD evaluation (matches real ONNX behavior)
    print(f"\n{'='*60}")
    print("FINAL RESULTS (evaluated with hard argmax)")
    print('='*60)
    p, i, d = best_params
    print(f"Best PID: p={float(p):.4f}, i={float(i):.4f}, d={float(d):.4f}")
    
    baseline_hard = eval_loss(jnp.array([0.195, 0.1, -0.053]), data)
    final_hard = eval_loss(best_params, data)
    
    print(f"\nBaseline cost (p=0.195, i=0.1, d=-0.053): {float(baseline_hard):.2f}")
    print(f"Best optimized cost: {float(final_hard):.2f}")
    improvement = (baseline_hard - final_hard) / baseline_hard * 100
    print(f"Improvement:    {float(improvement):.1f}%")


if __name__ == '__main__':
    main()
