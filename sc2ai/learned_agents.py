import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env.environment import StepType

from sc2ai.actor_critic import ConvActorCritic
import tensorflow as tf


class RoachesAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.discount_factor = 0.99
        state_shape = [7, 64, 64]
        self.action_options = [
            actions.FUNCTIONS.Attack_screen('now', [0, 0]),
            actions.FUNCTIONS.Attack_screen('now', [83, 0]),
            actions.FUNCTIONS.Attack_screen('now', [0, 83]),
            actions.FUNCTIONS.Attack_screen('now', [83, 83]),
            actions.FUNCTIONS.select_army('select')
        ]
        self.num_actions = len(self.action_options)
        self.actor_critic = ConvActorCritic(num_actions=self.num_actions, state_shape=state_shape)

        self.state_input = tf.placeholder(tf.float32, [None, *state_shape], name='state_input')
        self.action_mask_input = tf.placeholder(tf.float32, [None, self.num_actions], name='available_actions')
        self.discounted_reward_input = tf.placeholder(tf.float32, [None], name='discounted_rewards')
        self.actions_input = tf.placeholder(tf.int32, [None], name='actions')

        self.state_value = self.actor_critic.critic(self.state_input)
        self.actor_probs = self.actor_critic.actor(self.state_input, self.action_mask_input)

        self.loss_val = self.loss()
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss_val)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.states, self.actions, self.rewards, self.action_masks = [], [], [], []

    def loss(self):
        actions_one_hot = tf.one_hot(self.actions_input, self.num_actions)
        action_probs = tf.reduce_sum(self.actor_probs * actions_one_hot, axis=-1)
        advantage = self.discounted_reward_input - self.state_value

        entropy_bonus = tf.reduce_sum(0.1 * self.entropy(self.actor_probs))
        critic_loss = tf.reduce_sum(tf.square(advantage))
        actor_loss = -tf.reduce_sum(tf.log(action_probs) * tf.stop_gradient(advantage))
        return actor_loss - entropy_bonus + critic_loss

    @staticmethod
    def entropy(probs):
        return -tf.reduce_sum(probs * tf.log(probs + 1e-10), axis=-1)

    def get_action_mask(self, available_actions):
        mask = np.ones([self.num_actions])
        if actions.FUNCTIONS.Attack_screen.id not in available_actions:
            mask[:4] = 0
        return mask

    def step(self, obs):
        """
        :param obs: sc2 observation object
        :return: states, reward, done
        """
        super().step(obs)
        state = obs.observation.feature_minimap.astype(np.float32)

        if obs.step_type != StepType.FIRST:
            self.rewards.append(obs.reward)

        if obs.step_type != StepType.LAST:
            self.states.append(state)
            action_mask = self.get_action_mask(obs.observation.available_actions)
            action_probs = self.session.run(self.actor_probs, feed_dict={
                self.state_input: np.expand_dims(state, axis=0),
                self.action_mask_input: action_mask[np.newaxis]
            })
            chosen_action_index = np.random.choice(self.num_actions, p=action_probs[0])
            self.actions.append(chosen_action_index)
            self.action_masks.append(action_mask)
            return self.action_options[chosen_action_index]

    def discount(self, rewards):
        prev = 0
        discounted_rewards = np.copy(rewards)
        for i in range(1, len(discounted_rewards)):
            discounted_rewards[-i] += prev * self.discount_factor
            prev = discounted_rewards[-i]
        return discounted_rewards

    def train_policy(self):
        discounted_rewards = self.discount(self.rewards)
        loss, _ = self.session.run([self.loss_val, self.train_op], feed_dict={
            self.state_input: np.array(self.states),
            self.discounted_reward_input: discounted_rewards,
            self.actions_input: self.actions,
            self.action_mask_input: np.stack(self.action_masks)
        })
        print("Total reward: %.3f" % np.sum(self.rewards))
        print("Loss: %.3f" % loss)

    def reset(self):
        """
        Gets called after each episode. Training occurs here.
        """
        if len(self.states) != 0:
            self.train_policy()
        self.states, self.actions, self.rewards, self.action_masks = [], [], [], []
        super().reset()

    @staticmethod
    def _xy_locs(mask):
        """Mask should be a set of bools from comparison with a feature layer."""
        y, x = mask.nonzero()
        return list(zip(x, y))


































