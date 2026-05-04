import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_projection import continuous_action_to_discrete

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
        mask = self.env.unwrapped._last_action_mask
        return continuous_action_to_discrete(action, mask, self.grid_w, self.grid_h, self.num_rot)
