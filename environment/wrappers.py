import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ContinuousToDiscrete(gym.ActionWrapper):
    """
    Wraps a discrete placement environment to accept continuous actions in [-1, 1].
    Maps (x, y, rotation) from continuous space to grid indices, and automatically
    projects the action to the nearest valid placement to enforce DRC masking.
    """
    def __init__(self, env):
        super().__init__(env)
        # 3 dimensions: normalized X, normalized Y, normalized Rotation
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Grid parameters from the inner env
        self.grid_w = env.unwrapped.width
        self.grid_h = env.unwrapped.height
        self.num_rot = len(env.unwrapped.rotations)

    def action(self, action):
        # 1. Rescale [-1, 1] to [0, 1]
        x_norm, y_norm, rot_norm = (action + 1) / 2.0
        
        # 2. Map to indices
        px = int(np.clip(x_norm * self.grid_w, 0, self.grid_w - 1))
        py = int(np.clip(y_norm * self.grid_h, 0, self.grid_h - 1))
        prot = int(np.clip(rot_norm * self.num_rot, 0, self.num_rot - 1))
        
        # 3. Convert to flat index used by PCBEnv
        # Index = rotation * (W*H) + x * H + y  (based on _decode_action)
        flat_idx = prot * (self.grid_w * self.grid_h) + px * self.grid_h + py
        
        # 4. Project to nearest valid action using the environment's DRC mask
        mask = self.env.unwrapped._last_action_mask
        if not mask.any():
            return flat_idx
            
        if not mask[flat_idx]:
            # Reconstruct all valid (x, y, rot) and find the closest Euclidean distance
            valid_indices = np.where(mask)[0]
            valid_rot = valid_indices // (self.grid_w * self.grid_h)
            rem = valid_indices % (self.grid_w * self.grid_h)
            valid_x = rem // self.grid_h
            valid_y = rem % self.grid_h
            
            # Distance metric: prioritize spatial proximity over rotation
            dist = (valid_x - px)**2 + (valid_y - py)**2 + 0.5 * (valid_rot - prot)**2
            best_idx = valid_indices[np.argmin(dist)]
            return best_idx
            
        return flat_idx
