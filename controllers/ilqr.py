import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from . import BaseController
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from common import BINS, TEMPERATURE, MAX_ACC_DELTA, encode
from tinyphysics_eqx import create_model


# ============================================================================
# Dynamics
# ============================================================================

def make_dynamics_flat(model, temperature=TEMPERATURE):
    """Create a dynamics function: flat state R^41 x action R^1 -> flat next state R^41.

    State layout (41 dims):
      x[0:20]  = action_hist
      x[20:40] = lataccel_hist
      x[40]    = current_lataccel

    Exogenous inputs (not part of optimization state):
      exo_hist: [20, 3] = (roll, v, a) history
      exo_next: [3] = next step's (roll, v, a)
    """

    def dynamics_flat(x, u, exo_hist, exo_next):
        action_hist = x[:20]
        lataccel_hist = x[20:40]
        current_lataccel = x[40]

        # Clip action
        u_clipped = jnp.clip(u, -2.0, 2.0)

        # Shift histories, append new values
        action_hist_new = jnp.concatenate([action_hist[1:], u_clipped[None]])
        exo_hist_new = jnp.concatenate([exo_hist[1:], exo_next[None]], axis=0)

        # Build states tensor [1, 20, 4]: (action, roll, v, a)
        states = jnp.concatenate([
            action_hist_new[:, None],
            exo_hist_new,
        ], axis=-1)[None]

        # Hard-encode lataccel_hist to tokens (integer indexing = naturally zero-grad)
        tokens = encode(lataccel_hist)[None]  # [1, 20]

        # Use existing model forward pass
        logits = model(states, tokens)  # [1, 20, 1024]

        # EV decode: differentiable
        probs = jax.nn.softmax(logits[0, -1, :] / temperature)
        next_lataccel = jnp.dot(probs, BINS)

        # Rate limit
        next_lataccel = jnp.clip(next_lataccel, current_lataccel - MAX_ACC_DELTA,
                                  current_lataccel + MAX_ACC_DELTA)

        # Shift lataccel_hist, append new prediction
        lataccel_hist_new = jnp.concatenate([lataccel_hist[1:], next_lataccel[None]])

        # Pack into flat state
        x_next = jnp.concatenate([action_hist_new, lataccel_hist_new, next_lataccel[None]])
        return x_next

    return dynamics_flat


# ============================================================================
# Cost
# ============================================================================

def stage_cost(x_next, u, target, prev_lataccel, w_tracking=50.0, w_jerk=1.0, w_action=0.01, dt=0.1):
    """Per-step cost. x_next[40] = current_lataccel after this step."""
    lataccel = x_next[40]
    tracking = (lataccel - target) ** 2
    jerk = ((lataccel - prev_lataccel) / dt) ** 2
    action = u ** 2
    return w_tracking * tracking + w_jerk * jerk + w_action * action


# ============================================================================
# iLQR
# ============================================================================

def make_ilqr_solver(model, horizon=20, n_iters=5, temperature=TEMPERATURE,
                     w_tracking=50.0, w_jerk=1.0, w_action=0.01):
    """Build a JIT-compiled iLQR solver.

    Returns a function: ilqr_solve(x0, u_init, exo_hist, exo_future, targets) -> u_opt
    """
    dynamics = make_dynamics_flat(model, temperature)

    # Pre-compute jacobian function
    # A = df/dx [41, 41], B = df/du [41]
    jac_dynamics = jax.jacobian(dynamics, argnums=(0, 1))

    def cost_fn(x_next, u, target, prev_lataccel):
        return stage_cost(x_next, u, target, prev_lataccel, w_tracking, w_jerk, w_action)

    cost_grad_x = jax.grad(cost_fn, argnums=0)  # dl/dx_next [41]
    cost_grad_u = jax.grad(cost_fn, argnums=1)  # dl/du scalar
    cost_hess_xx = jax.hessian(cost_fn, argnums=0)  # d2l/dx2 [41,41]
    cost_hess_uu = jax.hessian(cost_fn, argnums=1)  # d2l/du2 scalar
    cost_hess_xu = jax.jacobian(jax.grad(cost_fn, argnums=0), argnums=1)  # d2l/dxdu [41]

    def forward_rollout(x0, u_seq, exo_hist, exo_future, targets):
        """Roll out dynamics, collect trajectory and costs.

        Args:
            x0: [41] initial flat state
            u_seq: [H] action sequence
            exo_hist: [20, 3] initial exo history
            exo_future: [H, 3] future exo values
            targets: [H] target lataccels

        Returns:
            x_traj: [H+1, 41] state trajectory
            exo_hist_traj: [H, 20, 3] exo history at each step
            total_cost: scalar
        """
        def step(carry, inputs):
            x, exo_h = carry
            u, exo_next, target = inputs

            prev_lataccel = x[40]
            x_next = dynamics(x, u, exo_h, exo_next)
            c = cost_fn(x_next, u, target, prev_lataccel)

            # Update exo_hist for next step
            exo_h_next = jnp.concatenate([exo_h[1:], exo_next[None]], axis=0)

            return (x_next, exo_h_next), (x_next, exo_h, c)

        (_, _), (x_traj_rest, exo_hist_traj, costs) = lax.scan(
            step, (x0, exo_hist), (u_seq, exo_future, targets)
        )

        x_traj = jnp.concatenate([x0[None], x_traj_rest], axis=0)  # [H+1, 41]
        total_cost = jnp.sum(costs)
        return x_traj, exo_hist_traj, total_cost

    def backward_pass(x_traj, u_seq, exo_hist_traj, exo_future, targets, mu):
        """Compute feedback gains via backward pass.

        Returns:
            k_seq: [H] feedforward gains (scalar per step)
            K_seq: [H, 41] feedback gains
        """
        H = horizon

        # Terminal value function (no terminal cost beyond last stage cost)
        V_x = jnp.zeros(41)
        V_xx = jnp.zeros((41, 41))

        def step(carry, inputs):
            V_x, V_xx = carry
            t = inputs  # time index (reversed)

            x_t = x_traj[t]
            u_t = u_seq[t]
            exo_h = exo_hist_traj[t]
            exo_next = exo_future[t]
            target = targets[t]
            prev_lataccel = x_traj[t, 40]  # current_lataccel before this step

            # Linearize dynamics
            A, B = jac_dynamics(x_t, u_t, exo_h, exo_next)
            # A: [41, 41], B: [41] (scalar control input)

            # Cost derivatives (w.r.t. x_next and u)
            x_next = x_traj[t + 1]
            l_x = cost_grad_x(x_next, u_t, target, prev_lataccel)  # [41]
            l_u = cost_grad_u(x_next, u_t, target, prev_lataccel)  # scalar
            l_xx = cost_hess_xx(x_next, u_t, target, prev_lataccel)  # [41, 41]
            l_uu = cost_hess_uu(x_next, u_t, target, prev_lataccel)  # scalar
            l_xu = cost_hess_xu(x_next, u_t, target, prev_lataccel)  # [41]

            # Q-function approximation
            Q_x = l_x + A.T @ V_x                          # [41]
            Q_u = l_u + B @ V_x                             # scalar
            Q_xx = l_xx + A.T @ V_xx @ A                    # [41, 41]
            Q_ux = l_xu @ A + B @ V_xx @ A                  # [41] (since u is scalar, Q_ux is [41])
            Q_uu = l_uu + B @ V_xx @ B                      # scalar

            # Regularize
            Q_uu_reg = Q_uu + mu

            # Gains (scalar action -> scalar division)
            k = -Q_u / Q_uu_reg                             # scalar feedforward
            K = -Q_ux / Q_uu_reg                            # [41] feedback

            # Update value function
            V_x_new = Q_x + K * Q_uu * k + K * Q_u + Q_ux * k  # [41]
            V_xx_new = Q_xx + jnp.outer(K, K) * Q_uu + jnp.outer(K, Q_ux) + jnp.outer(Q_ux, K)
            V_xx_new = (V_xx_new + V_xx_new.T) / 2  # symmetrize

            return (V_x_new, V_xx_new), (k, K)

        # Scan in reverse order
        indices = jnp.arange(H - 1, -1, -1)
        (_, _), (k_seq_rev, K_seq_rev) = lax.scan(step, (V_x, V_xx), indices)

        # Reverse to forward order
        k_seq = k_seq_rev[::-1]
        K_seq = K_seq_rev[::-1]
        return k_seq, K_seq

    def forward_with_gains(x0, u_seq, k_seq, K_seq, x_traj_nom, exo_hist, exo_future, targets, alpha):
        """Roll out with feedback gains applied."""
        def step(carry, inputs):
            x, exo_h = carry
            u_nom, k, K, x_nom, exo_next, target = inputs

            delta_x = x - x_nom
            u_new = u_nom + alpha * k + K @ delta_x
            u_new = jnp.clip(u_new, -2.0, 2.0)

            prev_lataccel = x[40]
            x_next = dynamics(x, u_new, exo_h, exo_next)
            c = cost_fn(x_next, u_new, target, prev_lataccel)

            exo_h_next = jnp.concatenate([exo_h[1:], exo_next[None]], axis=0)
            return (x_next, exo_h_next), (u_new, c)

        (_, _), (u_new_seq, costs) = lax.scan(
            step, (x0, exo_hist),
            (u_seq, k_seq, K_seq, x_traj_nom[:-1], exo_future, targets)
        )
        return u_new_seq, jnp.sum(costs)

    def ilqr_solve(x0, u_init, exo_hist, exo_future, targets):
        """Run iLQR optimization.

        Args:
            x0: [41] initial flat state
            u_init: [H] initial action sequence
            exo_hist: [20, 3] initial exo history
            exo_future: [H, 3] future exo values
            targets: [H] target lataccels

        Returns:
            u_opt: [H] optimized action sequence
        """
        alphas = jnp.array([1.0, 0.5, 0.25, 0.1])

        def iteration(carry, _):
            u_seq, mu, best_cost = carry

            # Forward rollout with current actions
            x_traj, exo_hist_traj, total_cost = forward_rollout(
                x0, u_seq, exo_hist, exo_future, targets)

            # Use the better of current rollout cost and previous best
            total_cost = jnp.minimum(total_cost, best_cost)

            # Backward pass
            k_seq, K_seq = backward_pass(
                x_traj, u_seq, exo_hist_traj, exo_future, targets, mu)

            # Line search
            def try_alpha(alpha):
                u_new, cost_new = forward_with_gains(
                    x0, u_seq, k_seq, K_seq, x_traj, exo_hist, exo_future, targets, alpha)
                return u_new, cost_new

            # Sequential line search: try each alpha, keep best
            def line_search_step(carry, alpha):
                best_u, best_c = carry
                u_new, cost_new = try_alpha(alpha)
                improved = cost_new < best_c
                best_u = jnp.where(improved, u_new, best_u)
                best_c = jnp.where(improved, cost_new, best_c)
                return (best_u, best_c), None

            (best_u, best_c), _ = lax.scan(
                line_search_step, (u_seq, total_cost), alphas)

            # Update regularization
            improved = best_c < total_cost
            mu_new = jnp.where(improved, jnp.maximum(mu / 10.0, 1e-6),
                                          jnp.minimum(mu * 10.0, 1e6))

            return (best_u, mu_new, best_c), None

        mu_init = 1.0
        init_cost = jnp.inf
        (u_opt, _, _), _ = lax.scan(iteration, (u_init, mu_init, init_cost), None, length=n_iters)
        return u_opt

    return jax.jit(ilqr_solve)


# ============================================================================
# Controller
# ============================================================================

class Controller(BaseController):
    def __init__(self):
        self.model = create_model('./models/tinyphysics.onnx')

        # iLQR parameters
        self.horizon = 20
        self.n_iters = 5
        self.temperature = TEMPERATURE

        # Build JIT-compiled solver
        self._solve = make_ilqr_solver(
            self.model,
            horizon=self.horizon,
            n_iters=self.n_iters,
            temperature=self.temperature,
        )

        # PID fallback
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0.0
        self.prev_error = 0.0

        # History state
        self.action_hist = None
        self.lataccel_hist = None
        self.exo_hist = None
        self.prev_u_seq = None
        self.step_count = 0
        self._jit_ready = False

    def _pid_action(self, target_lataccel, current_lataccel):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff

    def _bootstrap_histories(self):
        """Extract histories from simulator via stack inspection (first call only)."""
        import inspect
        frame = inspect.currentframe()
        try:
            frame = frame.f_back.f_back  # up through update() -> caller
            while frame:
                local_self = frame.f_locals.get('self', None)
                if local_self is not None and hasattr(local_self, 'action_history') and hasattr(local_self, 'current_lataccel_history'):
                    sim = local_self
                    n = len(sim.action_history)
                    if n >= 20:
                        self.action_hist = jnp.array(sim.action_history[-20:], dtype=jnp.float32)
                        self.lataccel_hist = jnp.array(
                            sim.current_lataccel_history[-20:], dtype=jnp.float32)
                        states = sim.state_history[-20:]
                        self.exo_hist = jnp.array(
                            [[s.roll_lataccel, s.v_ego, s.a_ego] for s in states],
                            dtype=jnp.float32)
                        return True
                frame = frame.f_back
        finally:
            del frame
        return False

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # PID always runs (for fallback and integral tracking)
        pid_action = self._pid_action(target_lataccel, current_lataccel)

        # Bootstrap or update histories
        if self.step_count == 0:
            if not self._bootstrap_histories():
                self.step_count += 1
                return np.clip(pid_action, -2, 2)
        else:
            # Shift histories with new data from simulator
            self.lataccel_hist = jnp.concatenate([
                self.lataccel_hist[1:],
                jnp.array([current_lataccel], dtype=jnp.float32)
            ])
            self.exo_hist = jnp.concatenate([
                self.exo_hist[1:],
                jnp.array([[state.roll_lataccel, state.v_ego, state.a_ego]], dtype=jnp.float32)
            ])
            # action_hist was already updated at end of previous update()

        # Build future trajectories
        H = self.horizon
        n_avail = len(future_plan.lataccel)
        H_use = min(H, n_avail)

        if H_use < H:
            # Pad if not enough future data
            exo_future = np.zeros((H, 3), dtype=np.float32)
            targets = np.zeros(H, dtype=np.float32)
            for t in range(H_use):
                exo_future[t] = [future_plan.roll_lataccel[t], future_plan.v_ego[t], future_plan.a_ego[t]]
                targets[t] = future_plan.lataccel[t]
            for t in range(H_use, H):
                exo_future[t] = exo_future[H_use - 1]
                targets[t] = targets[H_use - 1]
        else:
            exo_future = np.array([
                [future_plan.roll_lataccel[t], future_plan.v_ego[t], future_plan.a_ego[t]]
                for t in range(H)
            ], dtype=np.float32)
            targets = np.array([future_plan.lataccel[t] for t in range(H)], dtype=np.float32)

        exo_future = jnp.array(exo_future)
        targets = jnp.array(targets)

        # Build flat state
        x0 = jnp.concatenate([self.action_hist, self.lataccel_hist,
                               jnp.array([current_lataccel], dtype=jnp.float32)])

        # Initial action sequence (warm start or PID-based)
        if self.prev_u_seq is not None:
            u_init = self.prev_u_seq
        else:
            # Initialize with PID-like guess
            error = target_lataccel - current_lataccel
            u0 = np.clip(0.3 * error, -2, 2)
            u_init = jnp.full(H, u0, dtype=jnp.float32)

        # Run iLQR
        try:
            u_opt = self._solve(x0, u_init, self.exo_hist, exo_future, targets)
            has_nan = jnp.any(jnp.isnan(u_opt))
            if has_nan:
                if self.step_count < 5:
                    print(f"[iLQR] step {self.step_count}: NaN in u_opt, falling back to PID")
                action = float(pid_action)
            else:
                action = float(u_opt[0])
                if self.step_count < 5:
                    print(f"[iLQR] step {self.step_count}: u_opt[0]={action:.4f}, pid={pid_action:.4f}")

            # Warm start: shift for next call
            self.prev_u_seq = jnp.where(jnp.isnan(u_opt), 0.0, u_opt)
            self.prev_u_seq = jnp.concatenate([self.prev_u_seq[1:], self.prev_u_seq[-1:]])
        except Exception as e:
            if self.step_count < 5:
                print(f"[iLQR] step {self.step_count}: exception {e}, falling back to PID")
            action = float(pid_action)

        action = np.clip(action, -2, 2)

        # Update action_hist for next call
        self.action_hist = jnp.concatenate([
            self.action_hist[1:],
            jnp.array([action], dtype=jnp.float32)
        ])

        self.step_count += 1
        return action
