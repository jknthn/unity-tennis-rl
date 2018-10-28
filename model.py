import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FullyConnectedNetwork(nn.Module):
    
    def __init__(self, state_size, output_size, hidden_size, output_gate=None):
        super(FullyConnectedNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.output_gate = output_gate

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x


class PPOPolicyNetwork(nn.Module):
    
    def __init__(self, config):
        super(PPOPolicyNetwork, self).__init__()
        state_size = config['environment']['state_size']
        action_size = config['environment']['action_size']
        hidden_size = config['hyperparameters']['hidden_size']
        device = config['pytorch']['device']
        self.action_size = action_size

        self.actor_body = FullyConnectedNetwork(state_size, action_size, hidden_size, F.tanh)
        self.critic_body = FullyConnectedNetwork(state_size * 2.0 + action_size * 2.0, 1, hidden_size)  
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.to(device)

    def forward(self, local_observation, local_action=None, full_observation=None, full_action=None):
        local_obs = torch.Tensor(local_observation)
        full_observation = torch.Tensor(full_observation)

        a = self.actor_body(local_obs)

        v = None
        if full_observation is not None:
            full_obs = torch.Tensor(full_observation.view(full_observation.shape[0] * full_observation.shape[1], 1))

            d = torch.distributions.Normal(a, self.std)
            if full_action is None:
                full_action = d.sample()
            else:
                full_action = d.log_prob(full_action)
            print(full_action.shape)
            full_action = torch.cat((full_action, full_action), 0).view(-1, 1)
            print(full_action.shape)
            print(full_obs.shape)
            full_obs = torch.cat((full_obs, full_action), 0).view(1, -1)
            print(full_obs.shape)
            v = self.critic_body(full_obs)
        
        dist = torch.distributions.Normal(a, self.std)
        if local_action is None:
            local_action = dist.sample()
        log_prob = dist.log_prob(local_action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return local_action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), v
