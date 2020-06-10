import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun


def calibrated_init(size):
    return torch.from_numpy(np.random.randn(size[0], size[1]) / np.sqrt(size[1]))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, node1, node2):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.node1 = node1
        self.node2 = node2
        self.layer1 = nn.Linear(state_dim, self.node1)
        self.bn1 = nn.BatchNorm1d(num_features=self.node1)
        self.layer2 = nn.Linear(self.node1, self.node2)
        self.bn2 = nn.BatchNorm1d(num_features=self.node2)
        self.layer3 = nn.Linear(self.node2, action_dim)
        self.init_parameters()

    def init_parameters(self):
        self.layer1.weight.data = calibrated_init(self.layer1.weight.size())
        self.layer2.weight.data = calibrated_init(self.layer2.weight.size())
        self.layer3.weight.data = calibrated_init(self.layer3.weight.size())
        self.layer1.bias.data = torch.from_numpy(np.zeros(self.layer1.out_features))
        self.layer2.bias.data = torch.from_numpy(np.zeros(self.layer2.out_features))
        self.layer3.bias.data = torch.from_numpy(np.zeros(self.layer3.out_features))

    def forward(self, states):
        inter = self.layer1(states)
        inter = fun.relu(inter)
        inter = self.layer2(inter)
        inter = fun.relu(inter)
        inter = self.layer3(inter)
        proto_actions = torch.sigmoid(inter)
        return proto_actions


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, node1, node2):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.node1 = node1
        self.node2 = node2
        self.layer1 = nn.Linear(state_dim + action_dim, self.node1)
        self.bn1 = nn.BatchNorm1d(num_features=self.node1)
        self.layer2 = nn.Linear(self.node1, self.node2)
        self.bn2 = nn.BatchNorm1d(num_features=self.node2)
        self.layer3 = nn.Linear(self.node2, 1)
        self.init_parameters()

    def init_parameters(self):
        self.layer1.weight.data = calibrated_init(self.layer1.weight.size())
        self.layer2.weight.data = calibrated_init(self.layer2.weight.size())
        self.layer3.weight.data = calibrated_init(self.layer3.weight.size())
        self.layer1.bias.data = torch.from_numpy(np.zeros(self.layer1.out_features))
        self.layer2.bias.data = torch.from_numpy(np.zeros(self.layer2.out_features))
        self.layer3.bias.data = torch.from_numpy(np.zeros(self.layer3.out_features))

    def forward(self, states, actions):
        inter = torch.cat((states, actions), 1)
        inter = self.layer1(inter)
        inter = fun.relu(inter)
        inter = self.layer2(inter)
        inter = fun.relu(inter)
        q_values = self.layer3(inter)
        return q_values
