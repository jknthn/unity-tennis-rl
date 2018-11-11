import random
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from network import Network
from utils import ReplayBuffer, critic_input
from ddpgagent import DDPGAgent
from config import device


class MADDPGAgent():

    def __init__(self, config, file_prefix=None):

        self.buffer_size = config.hyperparameters.buffer_size
        self.batch_size = config.hyperparameters.batch_size
        self.update_frequency = config.hyperparameters.update_frequency
        self.gamma = config.hyperparameters.gamma
        self.number_of_agents = config.environment.number_of_agents
        self.noise_weight = config.hyperparameters.noise_start
        self.noise_decay = config.hyperparameters.noise_decay
        self.memory = ReplayBuffer(config)
        self.t = 0

        self.agents = [DDPGAgent(index, config) for index in range(self.number_of_agents)]

        if file_prefix:
            for i, to_load in enumerate(self.agents):
                f"{os.getcwd()}/models/by_score/{file_prefix}_actor_{i}.weights"
                actor_file = torch.load(f"{os.getcwd()}/models/by_score/{file_prefix}_actor_{i}.weights", map_location='cpu')
                critic_file = torch.load(f"{os.getcwd()}/models/by_score/{file_prefix}_critic_{i}.weights", map_location='cpu')
                to_load.actor_local.load_state_dict(actor_file)
                to_load.actor_target.load_state_dict(actor_file)
                to_load.critic_local.load_state_dict(critic_file)
                to_load.critic_target.load_state_dict(critic_file)
            print(f'Files loaded with prefix {file_prefix}')

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)
        all_next_states = all_next_states.reshape(1, -1)
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        self.t = (self.t + 1) % self.update_frequency
        if self.t == 0 and (len(self.memory) > self.batch_size):
            experiences = [self.memory.sample() for _ in range(self.number_of_agents)]
            self.learn(experiences, self.gamma)

    def act(self, all_states, add_noise=True, random=0.0):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, noise=self.noise_weight if add_noise else 0.0, random=random)
            self.noise_weight *= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)

    def learn(self, experiences, gamma):
        all_actions = []
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            all_actions.append(agent.actor_local(state))
            all_next_actions.append(agent.actor_target(next_state))
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)
