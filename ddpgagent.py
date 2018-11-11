import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from utils import critic_input, OUNoise
from network import Network
from config import device


class DDPGAgent():

    def __init__(self, index, config, filenames=None):
        random.seed(config.general.seed)
        np.random.seed(config.general.seed)

        self.noise = OUNoise(config)
        self.index = index
        self.action_size = config.environment.action_size
        self.tau = config.hyperparameters.tau

        self.actor_local = Network(config.actor, config.general.seed)
        self.actor_target = Network(config.actor, config.general.seed)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=config.actor.lr)
        self.critic_local = Network(config.critic, config.general.seed)
        self.critic_target = Network(config.critic, config.general.seed)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=config.critic.lr, weight_decay=config.hyperparameters.weight_decay)

    def act(self, state, noise, random):
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(torch.from_numpy(state).float().to(device)).cpu().data.numpy()
        self.actor_local.train()
        if noise is not None:
            action += self.noise.sample() * noise
        if random is not None:
            action = (1 - random) * action + random * (np.random.rand(self.action_size) - 0.5) * 2.0
        return np.clip(action, -1, 1)

    def learn(self, index, experiences, gamma, all_next_actions, all_actions):
        states, actions, rewards, next_states, dones = experiences

        self.critic_optimizer.zero_grad()

        index = torch.tensor([index]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            q_next = self.critic_target(critic_input(next_states, actions_next))
        q_exp = self.critic_local(critic_input(states, actions))
        q_t = rewards.index_select(1, index) + (gamma * q_next * (1 - dones.index_select(1, index)))
        F.mse_loss(q_exp, q_t.detach()).backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()

        actions_pred = [actions if i == self.index else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(critic_input(states, actions_pred)).mean()
        actor_loss.backward()

        self.actor_optimizer.step()

        self.actor_target.soft_update(self.actor_local, self.tau)
        self.critic_target.soft_update(self.critic_local, self.tau)
