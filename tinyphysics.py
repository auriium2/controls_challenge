import argparse
import importlib
import numpy as np
import onnxruntime as ort
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal
import urllib.request
import zipfile

from io import BytesIO
from collections import namedtuple
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import List, Union, Tuple, Dict
from tqdm.contrib.concurrent import process_map

from controllers import BaseController

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

DATASET_URL = "https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"
DATASET_PATH = Path(__file__).resolve().parent / "data"

class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value: Union[float, np.ndarray, List[float]]) -> Union[int, np.ndarray]:
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    return self.bins[token]

  def clip(self, value: Union[float, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
  def __init__(self, model_path: str, debug: bool) -> None:
    self.tokenizer = LataccelTokenizer()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    provider = 'CPUExecutionProvider'

    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, [provider])

  def softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def predict(self, input_data: dict, temperature=1.) -> Tuple[int, np.ndarray]:
    res = self.ort_session.run(None, input_data)[0]
    probs = self.softmax(res / temperature, axis=-1)
    # we only care about the last timestep (batch size is just 1)
    assert probs.shape[0] == 1
    assert probs.shape[2] == VOCAB_SIZE
    sample = np.random.choice(probs.shape[2], p=probs[0, -1])
    return sample, probs[0, -1]

  def predict_batch(self, input_data: dict, temperature=1., rngs: List = None) -> Tuple[np.ndarray, np.ndarray]:
    """Batched prediction - input_data has batch_size > 1
    
    Args:
      input_data: dict with 'states' and 'tokens' batched
      temperature: sampling temperature
      rngs: optional list of np.random.Generator for per-sample reproducibility
    """
    res = self.ort_session.run(None, input_data)[0]
    probs = self.softmax(res / temperature, axis=-1)
    # probs shape: (batch_size, seq_len, vocab_size)
    # we only care about the last timestep
    last_probs = probs[:, -1, :]  # (batch_size, vocab_size)
    # Sample for each item in batch
    if rngs is not None:
      samples = np.array([rng.choice(VOCAB_SIZE, p=p) for rng, p in zip(rngs, last_probs)])
    else:
      samples = np.array([np.random.choice(VOCAB_SIZE, p=p) for p in last_probs])
    return samples, last_probs

  def get_current_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> Tuple[float, np.ndarray]:
    tokenized_actions = self.tokenizer.encode(past_preds)
    raw_states = [list(x) for x in sim_states]
    states = np.column_stack([actions, raw_states])
    input_data = {
      'states': np.expand_dims(states, axis=0).astype(np.float32),
      'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
    }
    sample, probs = self.predict(input_data, temperature=0.8)
    return self.tokenizer.decode(sample), probs


class TinyPhysicsSimulator:
  def __init__(self, model: TinyPhysicsModel, data_path: str, controller: BaseController, debug: bool = False, collect_ev_diffs: bool = False, ev_mode: bool = False) -> None:
    self.data_path = data_path
    self.sim_model = model
    self.data = self.get_data(data_path)
    self.controller = controller
    self.debug = debug
    self.collect_ev_diffs = collect_ev_diffs
    self.ev_mode = ev_mode
    self.reset()

  def reset(self) -> None:
    self.step_idx = CONTEXT_LENGTH
    state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
    self.state_history = [x[0] for x in state_target_futureplans]
    self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
    self.current_lataccel_history = [x[1] for x in state_target_futureplans]
    self.target_lataccel_history = [x[1] for x in state_target_futureplans]
    self.target_future = None
    self.current_lataccel = self.current_lataccel_history[-1]
    self.current_probs = None  # Store latest softmax distribution for visualization
    seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)

  def get_data(self, data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
      'v_ego': df['vEgo'].values,
      'a_ego': df['aEgo'].values,
      'target_lataccel': df['targetLateralAcceleration'].values,
      'steer_command': -df['steerCommand'].values  # steer commands are logged with left-positive convention but this simulator uses right-positive
    })
    return processed_df

  def sim_step(self, step_idx: int) -> None:
    pred, probs = self.sim_model.get_current_lataccel(
      sim_states=self.state_history[-CONTEXT_LENGTH:],
      actions=self.action_history[-CONTEXT_LENGTH:],
      past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
    )
    self.current_probs = probs  # Store for visualization
    pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    if step_idx >= CONTROL_START_IDX:
      self.current_lataccel = pred
    else:
      self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

    self.current_lataccel_history.append(self.current_lataccel)

  def control_step(self, step_idx: int) -> None:
    action = self.controller.update(self.target_lataccel_history[step_idx], self.current_lataccel, self.state_history[step_idx], future_plan=self.futureplan)
    if step_idx < CONTROL_START_IDX:
      action = self.data['steer_command'].values[step_idx]
    action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
    self.action_history.append(action)

  def get_state_target_futureplan(self, step_idx: int) -> Tuple[State, float, FuturePlan]:
    state = self.data.iloc[step_idx]
    return (
      State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
      state['target_lataccel'],
      FuturePlan(
        lataccel=self.data['target_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
        roll_lataccel=self.data['roll_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
        v_ego=self.data['v_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
        a_ego=self.data['a_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist()
      )
    )

  def step(self) -> None:
    state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
    self.state_history.append(state)
    self.target_lataccel_history.append(target)
    self.futureplan = futureplan
    self.control_step(self.step_idx)
    self.sim_step(self.step_idx)
    self.step_idx += 1

  def plot_data(self, ax, lines, axis_labels, title) -> None:
    ax.clear()
    for line, label in lines:
      ax.plot(line, label=label)
    ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax.legend()
    ax.set_title(f"{title} | Step: {self.step_idx}")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

  def compute_cost(self) -> Dict[str, float]:
    target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

  def plot_softmax_distribution(self, ax) -> None:
    ax.clear()
    if self.current_probs is not None:
      lataccel_bins = self.sim_model.tokenizer.bins
      expected_value = np.sum(lataccel_bins * self.current_probs)
      ax.fill_between(lataccel_bins, self.current_probs, alpha=0.7)
      ax.axvline(self.current_lataccel, color='red', linestyle='--', linewidth=2, label=f'Sampled: {self.current_lataccel:.2f}')
      ax.axvline(expected_value, color='orange', linestyle='-', linewidth=2, label=f'EV: {expected_value:.2f}')
      ax.axvline(self.target_lataccel_history[-1], color='green', linestyle='--', linewidth=2, label=f'Target: {self.target_lataccel_history[-1]:.2f}')
      ax.set_xlim(LATACCEL_RANGE)
      ax.set_ylim(0, max(0.01, self.current_probs.max() * 1.1))
      ax.legend(loc='upper right')
    ax.set_title(f"Model Softmax Distribution | Step: {self.step_idx}")
    ax.set_xlabel('Lateral Acceleration')
    ax.set_ylabel('Probability')

  def rollout(self) -> Dict[str, float]:
    # Track EV vs sampled differences (for debug, collect_ev_diffs, or ev_mode)
    ev_sampled_diffs = []
    ev_history = []
    sampled_history = []
    
    if self.debug:
      plt.ion()
      fig, ax = plt.subplots(4, figsize=(12, 10), constrained_layout=True)
      # Store snapshots of distributions at specific timesteps
      snapshot_steps = [100, 200, 300, 400, 500]
      distribution_snapshots = {}
    
    if self.ev_mode:
      plt.ion()
      fig_ev, ax_ev = plt.subplots(3, figsize=(12, 10), constrained_layout=True)
      snapshot_steps = [100, 200, 300, 400, 500]
      distribution_snapshots = {}

    for _ in range(CONTEXT_LENGTH, len(self.data)):
      self.step()
      
      # Track EV vs sampled difference
      if (self.debug or self.collect_ev_diffs or self.ev_mode) and self.current_probs is not None:
        lataccel_bins = self.sim_model.tokenizer.bins
        ev = np.sum(lataccel_bins * self.current_probs)
        ev_sampled_diffs.append(self.current_lataccel - ev)
        ev_history.append(ev)
        sampled_history.append(self.current_lataccel)
      
      # Capture distribution snapshots
      if (self.debug or self.ev_mode) and self.step_idx in snapshot_steps and self.current_probs is not None:
        distribution_snapshots[self.step_idx] = {
          'probs': self.current_probs.copy(),
          'sampled': self.current_lataccel,
          'target': self.target_lataccel_history[-1]
        }
      
      if self.debug and self.step_idx % 10 == 0:
        print(f"Step {self.step_idx:<5}: Current lataccel: {self.current_lataccel:>6.2f}, Target lataccel: {self.target_lataccel_history[-1]:>6.2f}")
        self.plot_data(ax[0], [(self.target_lataccel_history, 'Target lataccel'), (self.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
        self.plot_data(ax[1], [(self.action_history, 'Action')], ['Step', 'Action'], 'Action')
        self.plot_data(ax[2], [(np.array(self.state_history)[:, 0], 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
        self.plot_softmax_distribution(ax[3])
        plt.pause(0.01)
      
      if self.ev_mode and self.step_idx % 10 == 0:
        # Plot 1: EV vs Sampled over time
        ax_ev[0].clear()
        steps = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + len(ev_history))
        ax_ev[0].plot(steps, sampled_history, label='Sampled', alpha=0.8)
        ax_ev[0].plot(steps, ev_history, label='EV', alpha=0.8)
        ax_ev[0].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5, label='Control Start')
        ax_ev[0].legend()
        ax_ev[0].set_title(f'EV vs Sampled Lateral Acceleration | Step: {self.step_idx}')
        ax_ev[0].set_xlabel('Step')
        ax_ev[0].set_ylabel('Lateral Acceleration')
        
        # Plot 2: Difference (Sampled - EV) over time
        ax_ev[1].clear()
        ax_ev[1].plot(steps, ev_sampled_diffs, color='purple', alpha=0.8)
        ax_ev[1].axhline(0, color='red', linestyle='--', alpha=0.5)
        ax_ev[1].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5, label='Control Start')
        ax_ev[1].set_title(f'Sampled - EV Difference | Step: {self.step_idx}')
        ax_ev[1].set_xlabel('Step')
        ax_ev[1].set_ylabel('Difference')
        
        # Plot 3: Current softmax distribution
        self.plot_softmax_distribution(ax_ev[2])
        plt.pause(0.01)

    if self.debug:
      plt.ioff()
      
      # Create distribution snapshots figure before showing
      if distribution_snapshots:
        n_snapshots = len(distribution_snapshots)
        fig2, axes2 = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4), constrained_layout=True)
        if n_snapshots == 1:
          axes2 = [axes2]
        
        lataccel_bins = self.sim_model.tokenizer.bins
        for i, (step, snapshot) in enumerate(sorted(distribution_snapshots.items())):
          ax2 = axes2[i]
          probs = snapshot['probs']
          expected_value = np.sum(lataccel_bins * probs)
          ax2.fill_between(lataccel_bins, probs, alpha=0.7)
          ax2.axvline(snapshot['sampled'], color='red', linestyle='--', linewidth=2, label=f'Sampled: {snapshot["sampled"]:.2f}')
          ax2.axvline(expected_value, color='orange', linestyle='-', linewidth=2, label=f'EV: {expected_value:.2f}')
          ax2.axvline(snapshot['target'], color='green', linestyle='--', linewidth=2, label=f'Target: {snapshot["target"]:.2f}')
          ax2.set_xlim(LATACCEL_RANGE)
          ax2.set_ylim(0, max(0.01, probs.max() * 1.1))
          ax2.set_title(f'Step {step}')
          ax2.set_xlabel('Lateral Acceleration')
          if i == 0:
            ax2.set_ylabel('Probability')
          ax2.legend(loc='upper right', fontsize=8)
        
        fig2.suptitle('Distribution Snapshots', fontsize=14)
      
      # Show EV vs Sampled analysis
      if ev_sampled_diffs:
        ev_sampled_diffs = np.array(ev_sampled_diffs)
        abs_diffs = np.abs(ev_sampled_diffs)
        
        # Print statistics
        print(f"\n{'='*50}")
        print("EV vs Sampled Analysis:")
        print(f"  Mean difference (sampled - EV):     {np.mean(ev_sampled_diffs):>8.4f}")
        print(f"  Std deviation of difference:        {np.std(ev_sampled_diffs):>8.4f}")
        print(f"  Mean absolute difference:           {np.mean(abs_diffs):>8.4f}")
        print(f"  Max absolute difference:            {np.max(abs_diffs):>8.4f}")
        print(f"  Worst case step:                    {np.argmax(abs_diffs) + CONTEXT_LENGTH}")
        print(f"  95th percentile absolute diff:      {np.percentile(abs_diffs, 95):>8.4f}")
        print(f"{'='*50}\n")
        
        # Create figure with histogram and time series
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        
        # Histogram of differences
        axes3[0].hist(ev_sampled_diffs, bins=50, alpha=0.7, edgecolor='black')
        axes3[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes3[0].axvline(np.mean(ev_sampled_diffs), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(ev_sampled_diffs):.4f}')
        axes3[0].set_xlabel('Sampled - EV')
        axes3[0].set_ylabel('Frequency')
        axes3[0].set_title(f'Distribution of (Sampled - EV)\nStd: {np.std(ev_sampled_diffs):.4f}, Max |diff|: {np.max(abs_diffs):.4f}')
        axes3[0].legend()
        
        # Time series of differences
        steps = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + len(ev_sampled_diffs))
        axes3[1].plot(steps, ev_sampled_diffs, alpha=0.7, linewidth=0.5)
        axes3[1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes3[1].fill_between(steps, -np.std(ev_sampled_diffs), np.std(ev_sampled_diffs), alpha=0.2, color='orange', label=f'±1 std ({np.std(ev_sampled_diffs):.4f})')
        worst_idx = np.argmax(abs_diffs)
        axes3[1].scatter([steps[worst_idx]], [ev_sampled_diffs[worst_idx]], color='red', s=100, zorder=5, label=f'Worst: {ev_sampled_diffs[worst_idx]:.4f}')
        axes3[1].set_xlabel('Step')
        axes3[1].set_ylabel('Sampled - EV')
        axes3[1].set_title('EV vs Sampled Difference Over Time')
        axes3[1].legend()
        
        fig3.suptitle('EV as Surrogate Analysis', fontsize=14)
      
      plt.show()
    
    if self.ev_mode:
      plt.ioff()
      
      ev_sampled_diffs = np.array(ev_sampled_diffs)
      abs_diffs = np.abs(ev_sampled_diffs)
      
      # Print statistics
      print(f"\n{'='*50}")
      print("EV vs Sampled Analysis:")
      print(f"  Mean difference (sampled - EV):     {np.mean(ev_sampled_diffs):>8.4f}")
      print(f"  Std deviation of difference:        {np.std(ev_sampled_diffs):>8.4f}")
      print(f"  Mean absolute difference:           {np.mean(abs_diffs):>8.4f}")
      print(f"  Max absolute difference:            {np.max(abs_diffs):>8.4f}")
      print(f"  Worst case step:                    {np.argmax(abs_diffs) + CONTEXT_LENGTH}")
      print(f"  95th percentile absolute diff:      {np.percentile(abs_diffs, 95):>8.4f}")
      print(f"{'='*50}\n")
      
      # Create distribution snapshots figure
      if distribution_snapshots:
        n_snapshots = len(distribution_snapshots)
        fig2, axes2 = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4), constrained_layout=True)
        if n_snapshots == 1:
          axes2 = [axes2]
        
        lataccel_bins = self.sim_model.tokenizer.bins
        for i, (step, snapshot) in enumerate(sorted(distribution_snapshots.items())):
          ax2 = axes2[i]
          probs = snapshot['probs']
          expected_value = np.sum(lataccel_bins * probs)
          ax2.fill_between(lataccel_bins, probs, alpha=0.7)
          ax2.axvline(snapshot['sampled'], color='red', linestyle='--', linewidth=2, label=f'Sampled: {snapshot["sampled"]:.2f}')
          ax2.axvline(expected_value, color='orange', linestyle='-', linewidth=2, label=f'EV: {expected_value:.2f}')
          ax2.set_xlim(LATACCEL_RANGE)
          ax2.set_ylim(0, max(0.01, probs.max() * 1.1))
          ax2.set_title(f'Step {step}')
          ax2.set_xlabel('Lateral Acceleration')
          if i == 0:
            ax2.set_ylabel('Probability')
          ax2.legend(loc='upper right', fontsize=8)
        
        fig2.suptitle('Distribution Snapshots', fontsize=14)
      
      # Create summary figure with histogram and final time series
      fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
      
      # Top left: EV vs Sampled over time
      steps = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + len(ev_history))
      axes3[0, 0].plot(steps, sampled_history, label='Sampled', alpha=0.8)
      axes3[0, 0].plot(steps, ev_history, label='EV', alpha=0.8)
      axes3[0, 0].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5, label='Control Start')
      axes3[0, 0].legend()
      axes3[0, 0].set_title('EV vs Sampled Lateral Acceleration')
      axes3[0, 0].set_xlabel('Step')
      axes3[0, 0].set_ylabel('Lateral Acceleration')
      
      # Top right: Difference over time
      axes3[0, 1].plot(steps, ev_sampled_diffs, color='purple', alpha=0.8)
      axes3[0, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
      axes3[0, 1].fill_between(steps, -np.std(ev_sampled_diffs), np.std(ev_sampled_diffs), alpha=0.2, color='orange', label=f'±1 std')
      worst_idx = np.argmax(abs_diffs)
      axes3[0, 1].scatter([steps[worst_idx]], [ev_sampled_diffs[worst_idx]], color='red', s=100, zorder=5, label=f'Worst: {ev_sampled_diffs[worst_idx]:.4f}')
      axes3[0, 1].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5)
      axes3[0, 1].legend()
      axes3[0, 1].set_title('Sampled - EV Difference Over Time')
      axes3[0, 1].set_xlabel('Step')
      axes3[0, 1].set_ylabel('Difference')
      
      # Bottom left: Histogram of differences
      axes3[1, 0].hist(ev_sampled_diffs, bins=50, alpha=0.7, edgecolor='black')
      axes3[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
      axes3[1, 0].axvline(np.mean(ev_sampled_diffs), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(ev_sampled_diffs):.4f}')
      axes3[1, 0].set_xlabel('Sampled - EV')
      axes3[1, 0].set_ylabel('Frequency')
      axes3[1, 0].set_title(f'Distribution of Differences\nStd: {np.std(ev_sampled_diffs):.4f}')
      axes3[1, 0].legend()
      
      # Bottom right: Histogram of absolute differences
      axes3[1, 1].hist(abs_diffs, bins=50, alpha=0.7, edgecolor='black', color='orange')
      axes3[1, 1].axvline(np.mean(abs_diffs), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(abs_diffs):.4f}')
      axes3[1, 1].axvline(np.percentile(abs_diffs, 95), color='purple', linestyle='--', linewidth=2, label=f'95th %ile: {np.percentile(abs_diffs, 95):.4f}')
      axes3[1, 1].set_xlabel('|Sampled - EV|')
      axes3[1, 1].set_ylabel('Frequency')
      axes3[1, 1].set_title(f'Absolute Difference Distribution\nMax: {np.max(abs_diffs):.4f}')
      axes3[1, 1].legend()
      
      fig3.suptitle('EV as Surrogate Analysis Summary', fontsize=14)
      
      # Create cumulative error bounds figure
      # This shows how per-step EV error could compound over the trajectory
      fig4, axes4 = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
      
      ev_history = np.array(ev_history)
      sampled_history = np.array(sampled_history)
      per_step_std = np.std(ev_sampled_diffs)
      
      # Cumulative error grows with sqrt(n) for independent errors (random walk)
      # Upper plot: Show EV trajectory with growing uncertainty bounds
      cumulative_steps = np.arange(len(ev_history))
      # Random walk std grows as std * sqrt(n)
      cumulative_std_random_walk = per_step_std * np.sqrt(cumulative_steps + 1)
      # Worst case: errors accumulate linearly
      cumulative_std_worst_case = per_step_std * (cumulative_steps + 1)
      
      axes4[0].plot(steps, ev_history, label='EV Trajectory', color='blue', linewidth=2)
      axes4[0].fill_between(steps, 
                            ev_history - cumulative_std_random_walk, 
                            ev_history + cumulative_std_random_walk, 
                            alpha=0.3, color='orange', label=f'±1σ Random Walk Bounds')
      axes4[0].fill_between(steps, 
                            ev_history - 2*cumulative_std_random_walk, 
                            ev_history + 2*cumulative_std_random_walk, 
                            alpha=0.15, color='orange', label=f'±2σ Random Walk Bounds')
      axes4[0].plot(steps, sampled_history, label='Actual Sampled', color='red', alpha=0.5, linewidth=1)
      axes4[0].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5)
      axes4[0].legend(loc='upper right')
      axes4[0].set_title(f'EV Trajectory with Cumulative Error Bounds\n(Per-step σ = {per_step_std:.4f}, assumes errors compound as random walk)')
      axes4[0].set_xlabel('Step')
      axes4[0].set_ylabel('Lateral Acceleration')
      
      # Lower plot: Show the growing uncertainty over time
      axes4[1].plot(steps, cumulative_std_random_walk, label='Random Walk (√n × σ)', color='orange', linewidth=2)
      axes4[1].plot(steps, cumulative_std_worst_case, label='Worst Case (n × σ)', color='red', linewidth=2, linestyle='--')
      axes4[1].axhline(per_step_std, color='blue', linestyle=':', label=f'Per-step σ = {per_step_std:.4f}')
      axes4[1].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5)
      axes4[1].legend()
      axes4[1].set_title('Cumulative Uncertainty Growth Over Rollout')
      axes4[1].set_xlabel('Step')
      axes4[1].set_ylabel('Cumulative Std Dev')
      axes4[1].set_ylim(0, min(cumulative_std_worst_case[-1], 5))  # Cap y-axis for readability
      
      fig4.suptitle('Potential Trajectory Divergence from EV Approximation', fontsize=14)
      
      plt.show()
    
    if self.debug or self.collect_ev_diffs or self.ev_mode:
      return self.compute_cost(), ev_sampled_diffs
    return self.compute_cost()


def get_available_controllers():
  return [f.stem for f in Path('controllers').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']


def run_rollout(data_path, controller_type, model_path, debug=False, collect_ev_diffs=False, ev_mode=False):
  tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
  controller = importlib.import_module(f'controllers.{controller_type}').Controller()
  sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=debug, collect_ev_diffs=collect_ev_diffs, ev_mode=ev_mode)
  result = sim.rollout()
  if debug or collect_ev_diffs or ev_mode:
    cost, ev_diffs = result
    return cost, sim.target_lataccel_history, sim.current_lataccel_history, ev_diffs
  return result, sim.target_lataccel_history, sim.current_lataccel_history, []


def run_batched_rollouts_for_ev(data_paths: List[Path], controller_type: str, model_path: str, batch_size: int = 32):
  """Run multiple rollouts in batched mode to collect EV diffs efficiently."""
  from tqdm import tqdm
  model = TinyPhysicsModel(model_path, debug=False)
  tokenizer = model.tokenizer
  ControllerClass = importlib.import_module(f'controllers.{controller_type}').Controller
  
  all_ev_diffs = []
  
  # Process in batches
  num_batches = (len(data_paths) + batch_size - 1) // batch_size
  for batch_start in tqdm(range(0, len(data_paths), batch_size), total=num_batches, desc='Batched rollouts'):
    batch_paths = data_paths[batch_start:batch_start + batch_size]
    current_batch_size = len(batch_paths)
    
    # Initialize all simulations in the batch with per-sim RNGs for reproducibility
    sims = []
    rngs = []
    for path in batch_paths:
      controller = ControllerClass()
      sim = TinyPhysicsSimulator(model, str(path), controller=controller, debug=False, collect_ev_diffs=False)
      sims.append(sim)
      # Create per-sim RNG with same seed logic as reset()
      seed = int(md5(str(path).encode()).hexdigest(), 16) % 10**4
      rngs.append(np.random.default_rng(seed))
    
    # Get the minimum length across all data files in batch
    min_len = min(len(sim.data) for sim in sims)
    
    # Run all simulations in lockstep with batched inference
    for step in range(CONTEXT_LENGTH, min_len):
      # Prepare batched inputs
      all_states = []
      all_tokens = []
      
      for sim in sims:
        # Get state/target/futureplan and update histories
        state, target, futureplan = sim.get_state_target_futureplan(step)
        sim.state_history.append(state)
        sim.target_lataccel_history.append(target)
        sim.futureplan = futureplan
        
        # Control step
        action = sim.controller.update(sim.target_lataccel_history[step], sim.current_lataccel, sim.state_history[step], future_plan=sim.futureplan)
        if step < CONTROL_START_IDX:
          action = sim.data['steer_command'].values[step]
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        sim.action_history.append(action)
        
        # Prepare model inputs
        tokenized_actions = tokenizer.encode(sim.current_lataccel_history[-CONTEXT_LENGTH:])
        raw_states = [list(x) for x in sim.state_history[-CONTEXT_LENGTH:]]
        states = np.column_stack([sim.action_history[-CONTEXT_LENGTH:], raw_states])
        
        all_states.append(states)
        all_tokens.append(tokenized_actions)
      
      # Batch inference
      batch_states = np.stack(all_states, axis=0).astype(np.float32)
      batch_tokens = np.stack(all_tokens, axis=0).astype(np.int64)
      input_data = {'states': batch_states, 'tokens': batch_tokens}
      
      samples, probs = model.predict_batch(input_data, temperature=0.8, rngs=rngs)
      
      # Update each simulation with its result
      for i, sim in enumerate(sims):
        pred = tokenizer.decode(samples[i])
        sim.current_probs = probs[i]
        
        # Compute EV diff
        lataccel_bins = tokenizer.bins
        ev = np.sum(lataccel_bins * probs[i])
        
        pred = np.clip(pred, sim.current_lataccel - MAX_ACC_DELTA, sim.current_lataccel + MAX_ACC_DELTA)
        if step >= CONTROL_START_IDX:
          sim.current_lataccel = pred
        else:
          sim.current_lataccel = sim.get_state_target_futureplan(step)[1]
        
        sim.current_lataccel_history.append(sim.current_lataccel)
        
        # Collect EV diff
        all_ev_diffs.append(sim.current_lataccel - ev)
      
      # Update step index
      for sim in sims:
        sim.step_idx = step + 1
  
  return all_ev_diffs


def download_dataset():
  print("Downloading dataset (0.6G)...")
  DATASET_PATH.mkdir(parents=True, exist_ok=True)
  with urllib.request.urlopen(DATASET_URL) as resp:
    with zipfile.ZipFile(BytesIO(resp.read())) as z:
      for member in z.namelist():
        if not member.endswith('/'):
          with z.open(member) as src, open(DATASET_PATH / os.path.basename(member), 'wb') as dest:
            dest.write(src.read())


if __name__ == "__main__":
  available_controllers = get_available_controllers()
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--debug", action='store_true', help='Debug mode with full visualization')
  parser.add_argument("--ev", action='store_true', help='EV analysis mode - shows EV vs sampled comparison')
  parser.add_argument("--cumulative", action='store_true', help='Collect EV vs sampled stats across many rollouts')
  parser.add_argument("--batch_size", type=int, default=32, help='Batch size for cumulative mode')
  parser.add_argument("--controller", default='pid', choices=available_controllers)
  args = parser.parse_args()

  if not DATASET_PATH.exists():
    download_dataset()

  data_path = Path(args.data_path)
  if data_path.is_file():
    cost, _, _, _ = run_rollout(data_path, args.controller, args.model_path, debug=args.debug, ev_mode=args.ev)
    print(f"\nAverage lataccel_cost: {cost['lataccel_cost']:>6.4}, average jerk_cost: {cost['jerk_cost']:>6.4}, average total_cost: {cost['total_cost']:>6.4}")
  elif data_path.is_dir():
    if args.cumulative:
      # Cumulative mode: collect EV vs sampled diffs across all rollouts (batched)
      files = sorted(data_path.iterdir())[:args.num_segs]
      print(f"Running batched rollouts with batch_size={args.batch_size}...")
      all_ev_diffs = run_batched_rollouts_for_ev(files, args.controller, args.model_path, batch_size=args.batch_size)
      all_ev_diffs = np.array(all_ev_diffs)
      abs_diffs = np.abs(all_ev_diffs)
      
      # Print statistics
      print(f"\n{'='*60}")
      print(f"Cumulative EV vs Sampled Analysis ({len(files)} rollouts, {len(all_ev_diffs)} samples)")
      print(f"  Mean difference (sampled - EV):     {np.mean(all_ev_diffs):>8.4f}")
      print(f"  Std deviation of difference:        {np.std(all_ev_diffs):>8.4f}")
      print(f"  Mean absolute difference:           {np.mean(abs_diffs):>8.4f}")
      print(f"  Max absolute difference:            {np.max(abs_diffs):>8.4f}")
      print(f"  95th percentile absolute diff:      {np.percentile(abs_diffs, 95):>8.4f}")
      print(f"  99th percentile absolute diff:      {np.percentile(abs_diffs, 99):>8.4f}")
      print(f"{'='*60}\n")
      
      # Plot histogram
      fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
      
      # Histogram of differences
      axes[0].hist(all_ev_diffs, bins=100, alpha=0.7, edgecolor='black')
      axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
      axes[0].axvline(np.mean(all_ev_diffs), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(all_ev_diffs):.4f}')
      axes[0].set_xlabel('Sampled - EV')
      axes[0].set_ylabel('Frequency')
      axes[0].set_title(f'Distribution of (Sampled - EV)\n{len(files)} rollouts, {len(all_ev_diffs)} samples')
      axes[0].legend()
      
      # Histogram of absolute differences
      axes[1].hist(abs_diffs, bins=100, alpha=0.7, edgecolor='black', color='orange')
      axes[1].axvline(np.mean(abs_diffs), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(abs_diffs):.4f}')
      axes[1].axvline(np.percentile(abs_diffs, 95), color='purple', linestyle='--', linewidth=2, label=f'95th %ile: {np.percentile(abs_diffs, 95):.4f}')
      axes[1].axvline(np.percentile(abs_diffs, 99), color='darkred', linestyle='--', linewidth=2, label=f'99th %ile: {np.percentile(abs_diffs, 99):.4f}')
      axes[1].set_xlabel('|Sampled - EV|')
      axes[1].set_ylabel('Frequency')
      axes[1].set_title(f'Absolute Difference Distribution\nStd: {np.std(all_ev_diffs):.4f}, Max: {np.max(abs_diffs):.4f}')
      axes[1].legend()
      
      fig.suptitle('Cumulative EV as Surrogate Analysis', fontsize=14)
      plt.show()
    else:
      run_rollout_partial = partial(run_rollout, controller_type=args.controller, model_path=args.model_path, debug=False)
      files = sorted(data_path.iterdir())[:args.num_segs]
      results = process_map(run_rollout_partial, files, max_workers=16, chunksize=10)
      costs = [result[0] for result in results]
      costs_df = pd.DataFrame(costs)
      print(f"\nAverage lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4}, average total_cost: {np.mean(costs_df['total_cost']):>6.4}")
      for cost in costs_df.columns:
        plt.hist(costs_df[cost], bins=np.arange(0, 1000, 10), label=cost, alpha=0.5)
      plt.xlabel('costs')
      plt.ylabel('Frequency')
      plt.title('costs Distribution')
      plt.legend()
      plt.show()
