import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env.environment import StepType
import time

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from pysc2.lib import features


class ConvActor(torch.nn.Module):
    def __init__(self, num_actions, screen_shape, state_shape):
        super().__init__()
        self.num_actions = num_actions
        self.state_shape = state_shape
        num_input_channels = state_shape[0]

        self.conv1 = torch.nn.Conv2d(num_input_channels, 32, 5, stride=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, stride=3)
        self.conv3 = torch.nn.Conv2d(32, 16, 3)
        self.linear1 = torch.nn.Linear(576, 32)
        self.head_non_spacial = torch.nn.Linear(32, self.num_actions)
        self.head_spacial_x = torch.nn.Linear(32, screen_shape[0])
        self.head_spacial_y = torch.nn.Linear(32, screen_shape[1])

    @staticmethod
    def sample(probs):
        categorical = Categorical(probs)
        chosen_action_index = categorical.sample()
        log_action_prob = categorical.log_prob(chosen_action_index)
        return chosen_action_index, log_action_prob

    def forward(self, state, action_mask):
        num_steps = state.shape[0]

        logits = F.relu(self.conv1(state))
        logits = F.relu(self.conv2(logits))
        logits = F.relu(self.conv3(logits))
        logits = logits.view(num_steps, -1)
        logits = F.relu(self.linear1(logits))

        non_spacial_probs = F.softmax(self.head_non_spacial(logits), dim=-1)
        spacial_x_probs = F.softmax(self.head_spacial_x(logits), dim=-1)
        spacial_y_probs = F.softmax(self.head_spacial_y(logits), dim=-1)

        masked_probs = (non_spacial_probs + 0.00001) * action_mask.type(torch.FloatTensor)
        masked_probs = masked_probs / masked_probs.sum()

        non_spacial_index, non_spacial_log_prob = self.sample(masked_probs)
        spacial_x, spacial_x_log_prob = self.sample(spacial_x_probs)
        spacial_y, spacial_y_log_prob = self.sample(spacial_y_probs)

        return non_spacial_index, spacial_x, spacial_y, \
            ((spacial_x_log_prob + spacial_y_log_prob) * int(non_spacial_index == 0) + non_spacial_log_prob)[0]


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
        return torch.squeeze(logits)


class RoachesAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.discount_factor = 0.99

        # This is the state shape for the mini-map, represented in channels_first order.
        state_shape = [2, 84, 84]

        self.num_actions = 2
        self.steps_per_action = 8

        self.resolution = 1
        screen_shape = (84 // self.resolution, 64 // self.resolution)

        self.actor = ConvActor(num_actions=self.num_actions, screen_shape=screen_shape, state_shape=state_shape)
        self.critic = ConvCritic(state_shape=state_shape)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.0005)

        # Define all input placeholders
        self.episode_counter = 0
        self.reward_buffer = self.step_index = 0
        self.states, self.rewards, self.log_action_probs = [], [], []
        self.load()

    def save(self):
        print('Saving weights')
        torch.save(self.actor.state_dict(), './weights/actor')
        torch.save(self.critic.state_dict(), './weights/critic')

    def load(self):
        print('Loading weights')
        self.actor.load_state_dict(torch.load('./weights/actor'))
        self.critic.load_state_dict(torch.load('./weights/critic'))

    def get_action_mask(self, available_actions):
        """
        Creates a mask array based on which actions are available

        :param available_actions: List of available action id's provided by pysc2
        :return: A 1 dimensional mask with 1 if an action is available and 0 if not.
        """
        mask = np.ones([self.num_actions])
        if actions.FUNCTIONS.Attack_screen.id not in available_actions:
            mask[0] = 0
        return torch.as_tensor(mask)

    def step(self, obs):
        """
        This function is called at each time step. At each step, we collect the (state, action, reward) tuple and
        save it for training.

        :param obs: sc2 observation object
        :return: states, reward, done
        """
        super().step(obs)
        time.sleep(0.2)
        self.reward_buffer += obs.reward

        player_relative = obs.observation.feature_screen.player_relative
        beacon = (np.array(player_relative) == 3).astype(np.float32)
        player = (np.array(player_relative) == 1).astype(np.float32)

        state = np.stack([beacon, player], axis=0)

        if (self.step_index % self.steps_per_action == 0 and obs.step_type != StepType.FIRST) \
                or obs.step_type == StepType.LAST:
            self.rewards.append(self.reward_buffer)
            self.reward_buffer = 0

        if self.step_index % self.steps_per_action == 0 and obs.step_type != StepType.LAST:
            self.states.append(state)
            action_mask = self.get_action_mask(obs.observation.available_actions)
            chosen_action_index, x, y, log_action_prob = self.actor(
                torch.as_tensor(np.expand_dims(state, axis=0)), action_mask.type(torch.FloatTensor))

            self.log_action_probs.append(log_action_prob)
            self.step_index += 1

            if chosen_action_index == 0:
                return actions.FUNCTIONS.Attack_screen('now', (x * self.resolution, y * self.resolution))
            else:
                return actions.FUNCTIONS.select_army('select')

        self.step_index += 1
        return actions.FUNCTIONS.no_op()

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

        # print("Total reward: %.3f" % np.sum(self.rewards))
        # print("Loss: %.3f" % loss_val.item())
        with open('rewards.txt', 'a+') as f:
            rewards = np.sum(self.rewards)
            f.write('%d\n' % rewards)

        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()

        if self.episode_counter % 50 == 0:
            self.save()
        self.episode_counter += 1

    def reset(self):
        """
        Gets called after each episode. Trains the agent and then resets all of the saved values.
        """
        if len(self.states) != 0:
            self.train_policy()
        self.step_index = 0
        self.states, self.rewards, self.log_action_probs = [], [], []
        super().reset()
<<<<<<< HEAD

    @staticmethod
    def _xy_locs(mask):
        """Mask should be a set of bools from comparison with a feature layer."""
        y, x = mask.nonzero()
        return list(zip(x, y))
=======
>>>>>>> b023ffb21f446fde35bfbe864a6a761b19d4be66
