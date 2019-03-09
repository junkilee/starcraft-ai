from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import sys

from sc2ai.tflearner.ac_network import ConvActorCritic
from sc2ai.tflearner.util import Util


class ActorCriticAgent(ABC):
    """ Agent that can provide all of the necessary tensors to train with Actor Critic using `ActorCriticLearner`.
    Training is not batched over games, so the tensors only need to provide outputs for a single trajectory.
    """
    @abstractmethod
    def step(self, states, masks):
        """
        Samples a batch of actions, given a batch of states and action masks.

        :param masks: Tensor of shape [batch_size, num_actions]. Mask contains 1 for available actions and 0 for
            unavailable actions.
        :param states: Tensor of shape [batch_size, *state_size]
        :return:
            action_indices:
                A list of indices that represents the chosen action at the state. This can be of any data type
                and will be passed back into self.get_feed_dict during training.
            actions:
                A list of pysc2 action indices, one for each state, that will be sent to the environment.
        """
        pass

    @abstractmethod
    def get_feed_dict(self, states, masks, actions, bootstrap_state):
        """
        Get the feed dict with values for all placeholders that are dependenceies for the tensors
        `bootstrap_value`, `train_values`, and `train_log_probs`.

        :param bootstrap_state: A numpy a array of shape [*state_shape] representing the terminal state.
        :param masks: A numpy array of shape [T, num_actions].
        :param states: A numpy array of shape [T, *state_shape].
        :param actions: A list of action indices with length T.
        :return: The feed dict required to evaluate `train_values` and `train_log_probs`
        """

    @abstractmethod
    def bootstrap_value(self):
        """
        :return: A scalar tensor representing the bootstrap value.
        """
        pass

    @abstractmethod
    def train_values(self):
        """
        :return: The tensor of shape [T] representing the estimated values of the states specified in self.get_feed_dict
        """
        pass

    @abstractmethod
    def train_log_probs(self):
        """
        :return:
            The tensor of shape [T] representing the log probability of performing the action in the state specified
            by the values in self.get_feed_dict
        """
        pass


class InterfaceAgent(ActorCriticAgent):
    def __init__(self, interface):
        self.interface = interface
        self.num_screen_dims = int(len(self.interface.screen_dimensions) / 2)

        tf.reset_default_graph()
        self.session = tf.Session()
        self.graph = tf.get_default_graph()

        self.state_input = tf.placeholder(tf.float32, [None, *self.interface.state_shape])  # [batch, *state_shape]
        self.bootstrap_state_input = tf.placeholder(tf.float32, [*self.interface.state_shape])

        self.mask_input = tf.placeholder(tf.float32, [None, self.interface.num_actions])  # [batch, num_actions]
        self.action_input = tf.placeholder(tf.int32, [None])   # [T]
        self.spacial_input = tf.placeholder(tf.int32, [None, 2])  # [T, 2]   dimension size 2 for x and y
        self.network = ConvActorCritic(self.interface.num_actions, self.num_screen_dims, self.interface.state_shape)

        # Tensor of shape [T, num_actions]
        self.nonspacial_probs = self.network.actor_nonspacial(self.state_input, self.mask_input)

        # List of length num screen dimensions of tensors of shape [T, screen_dimension]
        self.spacial_probs_x, self.spacial_probs_y = self.network.actor_spacial(
            self.state_input, self.interface.screen_dimensions)

    def step(self, state, mask):
        probs = self.session.run(
            [self.nonspacial_probs, *self.spacial_probs_x, *self.spacial_probs_y], {
                self.state_input: state,
                self.mask_input: mask
            })
        nonspacial_probs = probs[0]
        spacial_probs = probs[1:]
        spacial_probs_x = spacial_probs[:self.num_screen_dims]
        spacial_probs_y = spacial_probs[self.num_screen_dims:]  # [num_screen_dims, num_games, screen_dim]

        chosen_nonspacials = Util.sample_multiple(nonspacial_probs)  # [num_games]
        action_indices = []
        for i, chosen_nonspacial in enumerate(chosen_nonspacials):
            if chosen_nonspacial < self.num_screen_dims:
                x = np.random.choice(self.interface.screen_dimensions[chosen_nonspacial],
                                     p=spacial_probs_x[chosen_nonspacial][i])
                y = np.random.choice(self.interface.screen_dimensions[chosen_nonspacial],
                                     p=spacial_probs_y[chosen_nonspacial][i])
                action_indices.append((chosen_nonspacial, (x, y)))
            else:
                action_indices.append((chosen_nonspacial, None))
        return action_indices

    def get_feed_dict(self, states, masks, actions, bootstrap_state):
        nonspacial, spacial = zip(*actions)
        spacial = [(13, 27) if spacial is None else spacial for spacial in spacial]
        return {
            self.action_input: np.array(nonspacial),
            self.spacial_input: np.array(spacial),
            self.state_input: np.array(states),
            self.mask_input: np.array(masks),
            self.bootstrap_state_input: np.array(bootstrap_state)
        }

    def bootstrap_value(self):
        return tf.squeeze(self.network.critic(tf.expand_dims(self.bootstrap_state_input, axis=0)), axis=0)

    def train_values(self):
        return self.network.critic(self.state_input)

    def train_log_probs(self):
        nonspacial_log_probs = tf.log(Util.index(self.nonspacial_probs, self.action_input) + 0.00000001)

        # TODO: This only works if all screen dimensions are the same. Should pad to greatest length
        probs_y = self._get_chosen_spacial_prob(self.spacial_probs_y, self.spacial_input[:, 1])
        probs_x = self._get_chosen_spacial_prob(self.spacial_probs_x, self.spacial_input[:, 0])
        spacial_log_probs = tf.log(probs_x + 0.0000001) + tf.log(probs_y + 0.0000001)
        result = nonspacial_log_probs + tf.where(self.action_input < self.num_screen_dims,
                                                 x=spacial_log_probs,
                                                 y=tf.zeros_like(spacial_log_probs))
        return result

    def _get_chosen_spacial_prob(self, spacial_probs, spacial_choice):
        spacial_probs = tf.stack(spacial_probs, axis=-1)  # [T, screen_dim, num_screen_dimensions]
        spacial_probs = Util.index(spacial_probs, spacial_choice)  # [T, num_screen_dimensions]
        return Util.index(spacial_probs, self.action_input % tf.convert_to_tensor(self.num_screen_dims))  # [T]
























