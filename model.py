import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FullyConnectedNetwork(nn.Module):
    
    def __init__(self, state_size, output_size, hidden_size, output_gate=None):
        super(FullyConnectedNetwork, self).__init__()
        hidden_nodes, hidden_depth = hidden_size

        self.linear1 = nn.Linear(state_size, hidden_nodes)
        self.hidden = [nn.Linear(hidden_nodes, hidden_nodes) for _ in range(hidden_depth)]
        self.linear3 = nn.Linear(hidden_nodes, output_size)
        self.output_gate = output_gate

    def forward(self, x):
        x = F.relu(self.linear1(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x


class PPOPolicyNetwork(nn.Module):
    
    def __init__(self, config):
        super(PPOPolicyNetwork, self).__init__()
        number_of_agents = config['environment']['number_of_agents']
        state_size = config['environment']['state_size']
        action_size = config['environment']['action_size']
        a_hidden_size = config['hyperparameters']['a_hidden_size']
        c_hidden_size = config['hyperparameters']['c_hidden_size']
        device = config['pytorch']['device']

        self.action_size = action_size
        self.actors = []
        for i in range(number_of_agents):
            actor = FullyConnectedNetwork(state_size, action_size, a_hidden_size, F.tanh)
            self.actors.append(actor)
        self.critic_body = FullyConnectedNetwork(state_size, 1, c_hidden_size)  
        self.std = nn.Parameter(torch.ones(number_of_agents, action_size))
        self.to(device)

    def forward(self, obs, action=None):
        obs = torch.Tensor(obs) / 30.0
        a = torch.Tensor([])

        for i, actor in enumerate(self.actors):
            if len(obs.shape) == 3:
                actor_action = actor(obs[:,i,:]).view(obs.shape[0], 1, self.action_size)
                a = torch.cat((a, actor_action), 1)
            else:
                actor_action = actor(obs[i]).view(1, self.action_size)
                a = torch.cat((a, actor_action), 0)

        v = self.critic_body(obs)
        dist = torch.distributions.Normal(a, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        # print(action.shape, log_prob.shape, v.shape)
        return action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), v