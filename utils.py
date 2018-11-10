import random
import copy
from collections import namedtuple, deque

import numpy as np
import torch

from config import device


class OUNoise:

    def __init__(self, config):
        random.seed(config.general.seed)
        np.random.seed(config.general.seed)
        self.size = config.environment.action_size
        self.mu = config.noise.mu * np.ones(config.environment.action_size)
        self.theta = config.noise.theta
        self.sigma = config.noise.sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer():

    def __init__(self, config):
        random.seed(config.general.seed)
        np.random.seed(config.general.seed)
        self.action_size = config.environment.action_size
        self.memory = deque(maxlen=config.hyperparameters.buffer_size)
        self.batch_size = config.hyperparameters.batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def critic_input(states, actions):
    return torch.cat((states, actions), dim=1)


def hidden_init(layer):
    lim = 1. / np.sqrt(layer.weight.data.size()[0])
    return (-lim, lim)
