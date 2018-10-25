import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import units
from pysc2.lib import features

from sc2ai.actor_critic import BasicActorCritic
import tensorflow as tf


class RoachesAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        state_shape = [7, 64, 64]
        self.actions = [
            actions.FUNCTIONS.Attack_screen('now', [0, 0]),
            actions.FUNCTIONS.Attack_screen('now', [83, 0]),
            actions.FUNCTIONS.Attack_screen('now', [0, 83]),
            actions.FUNCTIONS.Attack_screen('now', [83, 83]),
            actions.FUNCTIONS.select_army('select')
        ]
        self.actor_critic = BasicActorCritic(num_actions=len(self.actions),
                                             state_shape=state_shape, architecture=(16,))
        self.state_input = tf.placeholder(tf.float32, state_shape, name='state_input')
        self.actor_probs = self.actor_critic.actor(self.state_input)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    @staticmethod
    def remove_unavailable_actions(observation, action_probs):
        if actions.FUNCTIONS.Attack_screen.id not in observation.available_actions:
            action_probs -= [1, 1, 1, 1, 0]
        return action_probs

    def step(self, obs):
        """
        :param obs: sc2 observation object
        :return: states, reward, done
        """
        super().step(obs)

        action_probs = self.session.run(self.actor_probs, feed_dict={
            self.state_input: obs.observation.feature_minimap.astype(np.float32)
        })
        # chosen_action_index = np.random.choice(len(self.actions), None, False, action_probs)
        chosen_action_index = np.random.choice(self.remove_unavailable_actions(obs.observation, action_probs))
        return self.actions[chosen_action_index]

    def reset(self):
        super().reset()

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

































