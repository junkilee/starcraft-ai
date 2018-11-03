import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env.environment import StepType

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from pysc2.lib import features


class SimpleActor(torch.nn.Module):
    def __init__(self, num_actions, state_shape):
        super().__init__()
        self.num_actions = num_actions
        self.state_shape = state_shape

        self.linear1 = torch.nn.Linear(np.prod(self.state_shape), 32)
        self.linear2 = torch.nn.Linear(32, self.num_actions)

    def forward(self, state, action_mask):
        logits = state.view(-1, np.prod(self.state_shape))
        logits = self.linear1(logits)
        logits = F.relu(logits)
        logits = self.linear2(logits)

        action_probs = F.softmax(logits, dim=-1)

        masked_probs = action_probs * action_mask.type(torch.FloatTensor)
        masked_probs = masked_probs / masked_probs.sum()
        categorical = Categorical(masked_probs)
        chosen_action_index = categorical.sample()
        log_action_prob = categorical.log_prob(chosen_action_index)
        return chosen_action_index, log_action_prob


class SimpleCritic(torch.nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.state_shape = state_shape

        self.linear1 = torch.nn.Linear(np.prod(self.state_shape), 32)
        self.linear2 = torch.nn.Linear(32, 1)

    def forward(self, state):
        logits = state.view(-1, np.prod(self.state_shape))
        logits = self.linear1(logits)
        logits = F.relu(logits)
        return self.linear2(logits)


class ConvActor(torch.nn.Module):
    def __init__(self, num_actions, state_shape):
        super().__init__()
        self.num_actions = num_actions
        self.state_shape = state_shape
        num_input_channels = state_shape[0]

        self.conv1 = torch.nn.Conv2d(num_input_channels, 32, 5, stride=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, stride=3)
        self.conv3 = torch.nn.Conv2d(32, 16, 3)

        self.linear1 = torch.nn.Linear(576, 32)
        self.linear2 = torch.nn.Linear(32, self.num_actions)

    def forward(self, state, action_mask):
        num_steps = state.shape[0]

        logits = F.relu(self.conv1(state))
        logits = F.relu(self.conv2(logits))
        logits = F.relu(self.conv3(logits))
        logits = logits.view(num_steps, -1)
        logits = F.relu(self.linear1(logits))
        logits = self.linear2(logits)
        action_probs = F.softmax(logits, dim=-1)

        masked_probs = action_probs * action_mask.type(torch.FloatTensor)
        masked_probs = masked_probs / masked_probs.sum()
        categorical = Categorical(masked_probs)
        chosen_action_index = categorical.sample()
        log_action_prob = categorical.log_prob(chosen_action_index)
        return chosen_action_index, log_action_prob


class ConvCritic(torch.nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.state_shape = state_shape
        num_input_channels = state_shape[0]

        self.conv1 = torch.nn.Conv2d(num_input_channels, 32, 5, stride=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, stride=3)
        self.conv3 = torch.nn.Conv2d(32, 16, 3)

        self.linear1 = torch.nn.Linear(576, 32)
        self.linear2 = torch.nn.Linear(32, 1)

    def forward(self, state):
        num_steps = state.shape[0]

        logits = F.relu(self.conv1(state))
        logits = F.relu(self.conv2(logits))
        logits = F.relu(self.conv3(logits))
        logits = logits.view(num_steps, -1)
        logits = F.relu(self.linear1(logits))
        logits = self.linear2(logits)
        return logits


class RoachesAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.discount_factor = 0.99

        # This is the state shape for the mini-map, represented in channels_first order.
        state_shape = [2, 84, 84]

        num_grid_points = (8, 6)  # Number of points going horizontally, vertically
        points = []
        for i in range(num_grid_points[0]):
            for j in range(num_grid_points[1]):
                points.append((i * int(83 / (num_grid_points[0] - 1)), j * int(63 / (num_grid_points[1] - 1))))

        # Available moves for agent include attack-moving into the corner.
        self.action_options = [
            actions.FUNCTIONS.select_army('select'),
        ]

        for point in points:
            self.action_options.append(actions.FUNCTIONS.Attack_screen('now', point))

        self.num_actions = len(self.action_options)

        self.actor = ConvActor(num_actions=self.num_actions, state_shape=state_shape)
        self.critic = ConvCritic(state_shape=state_shape)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.0005)

        # Define all input placeholders
        self.states, self.rewards, self.log_action_probs = [], [], []

    def get_action_mask(self, available_actions):
        """
        Creates a mask array based on which actions are available

        :param available_actions: List of available action id's provided by pysc2
        :return: A 1 dimensional mask with 1 if an action is available and 0 if not.
        """
        mask = np.ones([self.num_actions])
        if actions.FUNCTIONS.Attack_screen.id not in available_actions:
            mask[1:] = 0
        return torch.as_tensor(mask)

    def step(self, obs):
        """
        This function is called at each time step. At each step, we collect the (state, action, reward) tuple and
        save it for training.

        :param obs: sc2 observation object
        :return: states, reward, done
        """
        super().step(obs)

        player_relative = obs.observation.feature_screen.player_relative
        beacon = (np.array(player_relative) == 3).astype(np.float32)
        player = (np.array(player_relative) == 1).astype(np.float32)

        state = np.stack([beacon, player], axis=0)

        if obs.step_type != StepType.FIRST:
            self.rewards.append(obs.reward)

        if obs.step_type != StepType.LAST:
            self.states.append(state)
            action_mask = self.get_action_mask(obs.observation.available_actions)
            chosen_action_index, log_action_prob = self.actor(
                torch.as_tensor(np.expand_dims(state, axis=0)), action_mask.type(torch.FloatTensor))

            self.log_action_probs.append(log_action_prob)
            return self.action_options[chosen_action_index]

    def discount(self, rewards):
        """
        Computes sum of discounted rewards for each time step until the end of an episode.

        :param rewards: One dimensional array with the reward at each time step.
        :return: 1 dimensional array representing sum discounted rewards
        """
        prev = 0
        discounted_rewards = np.copy(rewards)
        for i in range(1, len(discounted_rewards)):
            discounted_rewards[-i] += prev * self.discount_factor
            prev = discounted_rewards[-i]

        return torch.as_tensor(np.array(discounted_rewards))

    def loss(self, states, discounted_rewards):
        advantage = discounted_rewards.type(torch.FloatTensor) - self.critic(states)
        actor_loss = -torch.stack(self.log_action_probs) * advantage.data
        critic_loss = advantage.pow(2)

        return actor_loss.mean() + 0.5 * critic_loss.mean()

    def train_policy(self):
        """
        Trains the policy on the saved (state, action, reward) tuple
        """
        discounted_rewards = self.discount(self.rewards)
        states = torch.as_tensor(np.array(self.states))
        loss_val = self.loss(states, discounted_rewards)

        print("Total reward: %.3f" % np.sum(self.rewards))
        print("Loss: %.3f" % loss_val.item())

        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()

    def reset(self):
        """
        Gets called after each episode. Trains the agent and then resets all of the saved values.
        """
        if len(self.states) != 0:
            self.train_policy()
        self.states, self.rewards, self.log_action_probs = [], [], []
        super().reset()


































