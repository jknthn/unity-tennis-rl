import random
import torch
import torch.nn.functional as F
import numpy as np


class NetworkConfig(object):

    def __init__(self, input_size, output_size, hidden_sizes=[512], lr=1e-3, output_activation=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.lr = lr
        self.output_activation = output_activation


class EnvironmentConfig(object):
    state_size = 24
    action_size = 2
    number_of_agents = 2


class GeneralConfig(object):
    seed = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HyperparametersConfig(object):
    tau = 1e-3
    weight_decay = 0.0
    buffer_size = int(1e5)
    batch_size = 128


class NoiseConfig(object):
    mu = 0
    theta = 0.15
    sigma = 0.2

class TrainingConfig(object):
    episode_count = 10000
    max_t = 1000
    solve_score = 0.5
    continue_after_solve = True

class Config(object):

    def __init__(self):
        self.environment = EnvironmentConfig()
        self.general = GeneralConfig()
        self.hyperparameters = HyperparametersConfig()
        self.noise = NoiseConfig()
        self.training = TrainingConfig()
        self.actor = NetworkConfig(
            input_size=self.environment.state_size,
            output_size=self.environment.action_size,
            hidden_sizes=[256, 256],
            lr=1e-3,
            output_activation=F.tanh
        )
        self.critic = NetworkConfig(
            input_size=(self.environment.state_size + self.environment.action_size) * self.environment.number_of_agents,
            output_size=1,
            hidden_sizes=[512],
            lr=1e-4,
            output_activation=None
        )


config = Config()
device = config.general.device