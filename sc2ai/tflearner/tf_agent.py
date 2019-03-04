from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import sys

from sc2ai.tflearner.actor_critic import ConvActorCritic
from .util import Util


class ActorCriticAgent(ABC):
    @abstractmethod
    def step(self, states, masks):
        """
        :param masks:
        :param states: A list of pysc2 timestep states from different trajectories.
        :return:
            action_indices:
                A list of indices that represents the chosen action at the state. This can be of any data type
                and will be passed back into self.get_feed_dict during training.
            actions:
                A list of pysc2 action generators, one for each state, that will be sent to the environment.
        """
        pass

    @abstractmethod
    def get_feed_dict(self, states, masks, actions):
        """
        :param states: A list of states with length T, the number of steps from a single trajectory.
        :param actions: A list of action indices with length T.
        :return: The feed dict required to evaluate self.train_values and self.train_log_probs
        """

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
        self.mask_input = tf.placeholder(tf.float32, [None, self.interface.num_actions])  # [batch, num_actions]
        self.action_input = tf.placeholder(tf.int32, [None])   # [T]
        self.spacial_input = tf.placeholder(tf.int32, [None, 2])  # [T, 2]   -  (x, y)
        self.network = ConvActorCritic(self.interface.num_actions, self.num_screen_dims, self.interface.state_shape)

        # Tensor of shape [T, num_actions]
        self.nonspacial_probs = self.network.actor_nonspacial(self.state_input, self.mask_input)

        # List of length num screen dimensions of tensors of shape [T, screen_dimension]
        self.spacial_probs_x, self.spacial_probs_y = self.network.actor_spacial(
            self.state_input, self.interface.screen_dimensions)

    def step(self, state, mask):
        # print("STEPPING")
        # print(mask)
        probs = self.session.run(
            [self.nonspacial_probs, *self.spacial_probs_x, *self.spacial_probs_y], {
                self.state_input: state,
                self.mask_input: mask
            })
        nonspacial_probs = probs[0]
        spacial_probs = probs[1:]
        spacial_probs_x = probs[:self.num_screen_dims]
        spacial_probs_y = probs[self.num_screen_dims:]
        # print(nonspacial_probs)

        chosen_nonspacials = Util.sample_multiple(nonspacial_probs)
        action_indices = []
        for chosen_nonspacial in chosen_nonspacials:
            if chosen_nonspacial < len(spacial_probs) / 2:
                x = Util.sample_multiple(spacial_probs_x[chosen_nonspacial])
                y = Util.sample_multiple(spacial_probs_y[chosen_nonspacial])
                action_indices.append((chosen_nonspacial, (x, y)))
            else:
                action_indices.append((chosen_nonspacial, None))
        return action_indices

    def get_feed_dict(self, states, masks, actions):
        nonspacial, spacial = zip(*actions)
        spacial = [(0, 0) if spacial is None else spacial for spacial in spacial]
        return {
            self.action_input: np.array(nonspacial),
            self.spacial_input: np.array(spacial),
            self.state_input: np.array(states),
            self.mask_input: np.array(masks)
        }

    def train_values(self):
        return self.network.critic(self.state_input)

    def train_log_probs(self):
        nonspacial_log_probs = tf.log(Util.index(self.nonspacial_probs, self.action_input) + 0.00000001)

        # TODO: This only works if all screen dimensions are the same. Should pad to greatest length
        probs_y = self._get_chosen_spacial_prob(self.spacial_probs_y, self.spacial_input[:, 1])
        probs_x = self._get_chosen_spacial_prob(self.spacial_probs_x, self.spacial_input[:, 0])
        spacial_log_probs = tf.log(probs_x + 0.0000001) + tf.log(probs_y + 0.0000001)

        # print_op = tf.print("log_prob debug:",
        #                     "\nnonspacial log probs", nonspacial_log_probs,
        #                     "\nspacial_log_probs", spacial_log_probs,
        #                     "\nspacial_probs_y", self.spacial_probs_y,
        #                     "\nprobs_y", probs_y,
        #
        #                     "\n\nspacial_probs_x", self.spacial_probs_x,
        #                     "\nprobs_x", probs_x,
        #                     output_stream=sys.stdout)

        # with tf.control_dependencies([print_op]):
        result = nonspacial_log_probs + tf.where(self.action_input < self.num_screen_dims,
                                                 x=spacial_log_probs,
                                                 y=tf.zeros_like(spacial_log_probs))
        return result

    def _get_chosen_spacial_prob(self, spacial_probs, nonspacial_choice):
        spacial_probs = tf.stack(spacial_probs, axis=-1)  # [T, screen_dim, num_screen_dimensions]
        spacial_probs = Util.index(spacial_probs, nonspacial_choice
                                   % tf.convert_to_tensor(self.num_screen_dims))  # [T, screen_dim]
        return Util.index(spacial_probs, self.action_input)  # [T]
























