In one step of the TinyPhysics simulator, the following detailed sequence occurs:

1. **Control Step** (`control_step()`):
   - The controller determines an action based on its type:
     - For "random" controller: `np.random.uniform(-1, 1)` generates a random value
     - For "zero" controller: returns `0.0`
   - The action is clipped to the range defined by `STEER_RANGE` (typically [-2, 2])
   - The action is stored in `action_history`

2. **Simulation Step** (`sim_step()`):
   - The method retrieves historical data:
     - `sim_states`: Last 20 states from `state_history`
     - `actions`: Last 20 actions from `action_history`
     - `past_preds`: Last 20 predictions from `current_lataccel_history`

   - It calls `get_current_lataccel()` which:
     a. Tokenizes the past predictions using `LataccelTokenizer.encode()`
     b. Combines states and actions into input arrays
     c. Prepares ONNX model input with proper shapes:
        - 'states': shape (1, 20, 4) containing [action, roll_lataccel, v_ego, a_ego]
        - 'tokens': shape (1, 20) containing encoded tokens
     d. Calls the model's `predict()` method with temperature=0.8
     e. Decodes the predicted token back to a float value using `LataccelTokenizer.decode()`

   - The prediction is clipped based on `MAX_ACC_DELTA` (0.5) to ensure smooth changes
   - If step index ≥ 100 (`CONTROL_START_IDX`), uses the prediction as current lataccel
   - Otherwise, uses target lataccel from `get_state_target_futureplan()`
   - Stores the current lataccel in `current_lataccel_history`

3. **State Update** (in `step()`):
   - The state is updated based on the current lataccel using physics simulation logic
   - The new state is stored in `state_history`
   - The step count is incremented

This process repeats for each simulation step, with the model making predictions based on historical data and the controller providing steering inputs that influence these predictions through the physics simulation.

```
import inspect

class TransformerController(BaseController):
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Get the current call stack
        frame = inspect.currentframe()
        try:
            # Walk up the stack to find the simulator
            while frame:
                if 'self' in frame.f_locals:
                    sim_self = frame.f_locals['self']
                    # Check if this looks like a TinyPhysicsSimulator
                    if hasattr(sim_self, 'action_history'):
                        # Found it! Access the action history
                        action_history = sim_self.action_history[-20:]  # Get last 20 steps
                        break
                frame = frame.f_back
        finally:
            # Clean up to avoid reference cycles
            del frame

        # Now use action_history for your Jacobian computation...
```

import inspect

class TransformerController(BaseController):
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Get simulator instance via call stack inspection
        frame = inspect.currentframe()
        try:
            while frame:
                if 'self' in frame.f_locals:
                    sim_self = frame.f_locals['self']
                    if hasattr(sim_self, 'action_history'):
                        # Found the simulator! Get all histories
                        state_history = sim_self.state_history[-20:]  # Last 20 states
                        action_history = sim_self.action_history[-20:]  # Last 20 actions
                        lataccel_history = sim_self.current_lataccel_history[-20:]  # Last 20 predictions
                        target_history = sim_self.target_lataccel_history[-20:]  # Last 20 targets

                        # Now bootstrap your local transformer
                        jacobian = self.compute_dynamics_jacobian(
                            state_history,
                            action_history,
                            lataccel_history,
                            target_history
                        )
                        break
                frame = frame.f_back
        finally:
            del frame  # Clean up

        # Continue with your controller logic...

import inspect

class TransformerController(BaseController):
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Get simulator instance via call stack inspection
        frame = inspect.currentframe()
        try:
            while frame:
                if 'self' in frame.f_locals:
                    sim_self = frame.f_locals['self']
                    if hasattr(sim_self, 'action_history'):
                        # Found the simulator! Get all histories
                        state_history = sim_self.state_history[-20:]  # Last 20 states
                        action_history = sim_self.action_history[-20:]  # Last 20 actions
                        lataccel_history = sim_self.current_lataccel_history[-20:]  # Last 20 predictions
                        target_history = sim_self.target_lataccel_history[-20:]  # Last 20 targets

                        # Now bootstrap your local transformer
                        jacobian = self.compute_dynamics_jacobian(
                            state_history,
                            action_history,
                            lataccel_history,
                            target_history
                        )
                        break
                frame = frame.f_back
        finally:
            del frame  # Clean up

        # Continue with your controller logic...
