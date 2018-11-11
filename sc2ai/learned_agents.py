import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env.environment import StepType
import torch
from sc2ai.actor_critic import ConvActorCritic


class RoachesAgent(base_agent.BaseAgent):
    def __init__(self, model, use_cuda):
        super().__init__()
        self.discount_factor = 0.99

        use_cuda = False
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        # This is the state shape for the mini-map, represented in channels_first order.
        state_shape = [2, 84, 84]
        self.num_actions = 2
        screen_shape = (84, 64)

        # self.actor = ConvActorCritic(num_actions=self.num_actions,
        #                              screen_shape=screen_shape,
        #                              state_shape=state_shape).to(device)
        self.actor = model
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0005)

        # Define all input placeholders
        self.episode_counter = 0
        self.reward_buffer = self.step_index = 0
        self.rewards, self.log_action_probs, self.entropys, self.critic_values = [], [], [], []
        # self.load()

    def save(self):
        print('Saving weights')
        torch.save(self.actor.state_dict(), './weights/actor')

    def load(self):
        print('Loading weights')
        self.actor.load_state_dict(torch.load('./weights/actor'))

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

        player_relative = obs.observation.feature_screen.player_relative
        beacon = (np.array(player_relative) == 3).astype(np.float32)
        player = (np.array(player_relative) == 1).astype(np.float32)

        state = np.stack([beacon, player], axis=0)

        if obs.step_type != StepType.FIRST:
            self.rewards.append(obs.reward)

        if obs.step_type != StepType.LAST:
            action_mask = self.get_action_mask(obs.observation.available_actions)
            chosen_action_index, x, y, entropy, log_action_prob, critic_value = self.actor(
                torch.as_tensor(np.expand_dims(state, axis=0), device=self.device),
                action_mask.type(self.dtype))

            self.entropys.append(entropy)
            self.critic_values.append(critic_value)
            self.log_action_probs.append(log_action_prob)

            if chosen_action_index == 0:
                return actions.FUNCTIONS.Attack_screen('now', (x, y))
            else:
                return actions.FUNCTIONS.select_army('select')

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

        return torch.as_tensor(np.array(discounted_rewards), device=self.device)

    def loss(self, discounted_rewards):
        advantage = discounted_rewards.type(self.dtype) - torch.stack(self.critic_values)
        actor_loss = -torch.stack(self.log_action_probs).type(self.dtype) * advantage.data
        critic_loss = advantage.pow(2)

        return actor_loss.mean() + 0.5 * critic_loss.mean() - 0.01 * torch.stack(self.entropys).mean()

    def train_policy(self):
        """
        Trains the policy on the saved (state, action, reward) tuple
        """
        discounted_rewards = self.discount(self.rewards)
        loss_val = self.loss(discounted_rewards)

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
        if len(self.rewards) != 0:
            self.train_policy()
        self.step_index = 0
        self.rewards, self.log_action_probs, self.entropys, self.critic_values = [], [], [], []
        super().reset()
