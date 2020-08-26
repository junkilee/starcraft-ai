import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from sc2ai.spinup.algorithms.ppo.core import mlp
from sc2ai.envs.actions import ActionVectorType
import math


class SC2Actor(nn.Module):
    def __init__(self, previous_modules, hidden_units, action_spec, device):
        super().__init__()
        self._previous_modules = previous_modules
        self._hidden_units = hidden_units
        self._action_spec = action_spec
        self.logit_nets = list()
        self.device = device
        self.to(device)

        print("action register-----------------------------")
        for action_tuple in self._action_spec:
            print(action_tuple)
            if action_tuple[0] is ActionVectorType.ACTION_TYPE:
                self.logit_nets += [
                    torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, action_tuple[1])).to(device)]
            elif action_tuple[0] is ActionVectorType.SCALAR:
                self.logit_nets += [
                    torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, action_tuple[1])).to(device)]
            elif action_tuple[0] is ActionVectorType.SPATIAL:
                xy_size = int(math.sqrt(action_tuple[1]))
                self.logit_nets += \
                    [(torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, xy_size)).to(device),
                      torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, xy_size)).to(device))]
            else:
                raise Exception("Such ActionVectorType is not defined.")
        print("action register-----------------------------")

    def distributions(self, obs):
        distributions = []
        _id = 0
        for action_tuple in self._action_spec:
            if action_tuple[0] is ActionVectorType.ACTION_TYPE or action_tuple[0] is ActionVectorType.SCALAR:
                distributions += [Categorical(logits=self.logit_nets[_id](obs))]
            elif action_tuple[0] is ActionVectorType.SPATIAL:
                distributions += [(Categorical(logits=self.logit_nets[_id][0](obs)),
                                   Categorical(logits=self.logit_nets[_id][1](obs)))]
            else:
                raise Exception("Such ActionVectorType is not defined.")
            _id += 1
        return distributions

    def log_prob_from_distributions(self, pis, acts):
        log_probs = list()
        _id = 0
        for distribution in pis:
            action_tuple = self._action_spec[_id]
            if isinstance(distribution, tuple):
                xy_size = int(math.sqrt(action_tuple[1]))
                #print(act[_id], xy_size)
                log_probs.append(distribution[0].log_prob(acts[:, _id] // xy_size) +
                                 distribution[1].log_prob(acts[:, _id] % xy_size))
            else:
                log_probs.append(distribution.log_prob(acts[:, _id]))
            _id += 1
        return torch.sum(torch.stack(log_probs, 1), 1)

    def sample(self, pis):
        a_vector = list()
        _id = 0
        for distribution in pis:
            action_tuple = self._action_spec[_id]
            if isinstance(distribution, tuple):
                xy_size = int(math.sqrt(action_tuple[1]))
                a_vector.append(distribution[0].sample() * xy_size + distribution[1].sample())
            else:
                a_vector.append(distribution.sample())
            _id += 1
        return torch.stack(a_vector, 1)

    def forward(self, obs, act=None):
        pis = self.distributions(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_distributions(pis, act)
        return pis, logp_a


class SC2Critic(nn.Module):
    def __init__(self, previous_modules, hidden_units, device):
        super().__init__()
        self.v_net = torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, 1)).to(device)
        self.device = device

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has the right shape.


class SC2AtariNetActorCritic(nn.Module):
    def __init__(self, observation_space, action_spec=None, hidden_units=256, activation=nn.ReLU,
                 device=torch.device('cpu')):
        super().__init__()
        self.conv1 = nn.Conv2d(observation_space['feature_screen'].shape[0], 16, 8, stride=4).to(device)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2).to(device)  # 32 channel 9,9
        self.convs_sequence = nn.Sequential(self.conv1,
                                            activation(),
                                            self.conv2,
                                            activation(),
                                            nn.Flatten(),
                                            nn.Linear(32 * 9 * 9, hidden_units),
                                            nn.ReLU()).to(device)
        self.device = device
        self.pi = SC2Actor(self.convs_sequence, hidden_units, action_spec, device)
        self.pi.to(device=device)
        self.v = SC2Critic(self.convs_sequence, hidden_units, device)
        self.v.to(device=device)
        self.to(device=device)

    def step(self, obs):
        with torch.no_grad():
            pis = self.pi.distributions(obs)
            a = self.pi.sample(pis)
            logp_a = self.pi.log_prob_from_distributions(pis, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

# class SC2FullyConvActorCritic(nn.Module):


# class SC2FullyConvLSTMActorCritic(nn.Module):
