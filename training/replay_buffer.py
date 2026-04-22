import random
import torch
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('spatial_obs', 'graph_data', 'action', 'next_spatial_obs', 'reward', 'done'))

class GraphReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return transitions

    def __len__(self):
        return len(self.memory)
