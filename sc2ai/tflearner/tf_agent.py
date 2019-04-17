from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

from sc2ai.env_interface import ActionParamType, AgentAction
from sc2ai.tflearner import util
from sc2ai.tflearner import parts
from sc2ai.tflearner.attention import SelfAttention


class ActorCriticAgent(ABC):
    """ Agent that can provide all of the necessary tensors to train with Actor Critic using `ActorCriticLearner`.
    Training is not batched over games, so the tensors only need to provide outputs for a single trajectory.
    """

    def __init__(self):
        tf.reset_default_graph()
        self.session = tf.Session()
        self.graph = tf.get_default_graph()

    @abstractmethod
    def step(self, states, masks, memory):
        """
        Samples a batch of actions, given a batch of states and action masks.

        :param memory: Memory returned by the previous step, or None for the first step.
        :param masks: Tensor of shape [batch_size, num_actions]. Mask contains 1 for available actions and 0 for
            unavailable actions.
        :param states: Tensor of shape [batch_size, *state_size]
        :return:
            action_indices:
                A list of AgentActions. This will be passed back into self.get_feed_dict during training.
            memory:
                An arbitrary object that will be passed into step at the next timestep.
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

    @abstractmethod
    def get_feed_dict(self, states, memory, masks, actions):
        """
        Get the feed dict with values for all placeholders that are dependenceies for the tensors
        `bootstrap_value`, `train_values`, and `train_log_probs`.

        :param memory: Memory corresponding to the state.
        :param masks: A numpy array of shape [T, num_actions].
        :param states: A numpy array of shape [T, *state_shape].
        :param actions: A list of action indices with length T.
        :return: The feed dict required to evaluate `train_values` and `train_log_probs`
        """

    def get_initial_memory(self, num_agents):
        return [None] * num_agents


class InterfaceAgent(ActorCriticAgent, ABC):
    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.num_actions = self.interface.num_actions()
        self.num_spatial_actions = self.interface.num_spatial_actions()
        self.num_select_actions = self.interface.num_select_unit_actions()

        self.state_input = tf.placeholder(tf.float32, [None, *self.interface.state_shape], name='state_input')  # [batch, *state_shape]
        self.mask_input = tf.placeholder(tf.float32, [None, self.interface.num_actions()], name='mask_input')  # [batch, num_actions]

        self.action_input = tf.placeholder(tf.int32, [None], name='action_input')  # [T]
        self.spacial_input = tf.placeholder(tf.int32, [None, 2], name='spatial_input')  # [T, 2]   dimension size 2 for x and y
        self.unit_selection_input = tf.placeholder(tf.int32, [None], name="unit_selection_input")

    def get_feed_dict(self, states, memory, masks, actions=None):
        feed_dict = {
            self.state_input: np.array(states),
            self.mask_input: np.array(masks),
        }
        if actions is not None:
            nonspatial, spacial, _, _ = zip(*[a.as_tuple() for a in actions])
            spacial = [(-1, -1) if spacial is None else spacial for spacial in spacial]
            feed_dict[self.action_input] = np.array(nonspatial)
            feed_dict[self.spacial_input] = np.array(spacial)
        return feed_dict

    def _get_chosen_selection_probs(self, selection_probs, selection_choice):
        """
        :param selection_probs: Tensor of integers of shape [T, num_units, num_selection_actions]
        :param selection_choice: Tensor of shape [T] of type int
        :return:
        """
        selection_probs = util.index(selection_probs, selection_choice)  # [T, num_selection_actions]
        num_selection_actions = self.interface.num_select_unit_actions()

        index = (self.action_input - self.num_spatial_actions) % tf.convert_to_tensor(num_selection_actions)
        return util.index(selection_probs, index)  # [T]

    def _get_chosen_spacial_prob(self, spacial_probs, spacial_choice):
        spacial_probs = util.index(spacial_probs, spacial_choice)  # [T, num_screen_dimensions]
        return util.index(spacial_probs, self.action_input % tf.convert_to_tensor(self.num_spatial_actions))  # [T]

    def _train_log_probs(self, nonspatial_probs, spatial_probs=None, selection_probs=None):
        nonspatial_log_probs = tf.log(util.index(nonspatial_probs, self.action_input) + 1e-10)

        result = nonspatial_log_probs
        if spatial_probs is not None:
            probs_y = self._get_chosen_spacial_prob(spatial_probs[0], self.spacial_input[:, 1])
            probs_x = self._get_chosen_spacial_prob(spatial_probs[1], self.spacial_input[:, 0])
            spacial_log_probs = tf.log(probs_x + 1e-10) + tf.log(probs_y + 1e-10)
            result = result + tf.where(self.action_input < self.num_spatial_actions,
                                       x=spacial_log_probs,
                                       y=tf.zeros_like(spacial_log_probs))

        if selection_probs is not None:
            probs_selection = self._get_chosen_selection_probs(selection_probs, self.unit_selection_input)
            selection_log_prob = tf.log(probs_selection + 1e-10)
            is_select_action = tf.logical_and(self.action_input >= self.num_spatial_actions,
                                              self.action_input < self.num_spatial_actions + self.num_select_actions)
            result = result + tf.where(is_select_action,
                                       x=selection_log_prob,
                                       y=tf.zeros_like(selection_log_prob))
        return result

    def _probs_from_features(self, features):
        num_steps = tf.shape(self.mask_input)[0]
        nonspatial_probs = parts.actor_nonspatial_head(features[:num_steps], self.mask_input, self.num_actions)
        spatial_probs = parts.actor_spatial_head(features[:num_steps], screen_dim=84, num_spatial_actions=self.num_spatial_actions)
        return nonspatial_probs, spatial_probs


class ConvAgent(InterfaceAgent):
    def __init__(self, interface):
        super().__init__(interface)
        self.features = parts.conv_body(self.state_input)
        self.nonspatial_probs, self.spatial_probs = self._probs_from_features(self.features)

    def step(self, state, mask, memory):
        nonspatial_probs, spatial_probs = self.session.run(
            [self.nonspatial_probs, self.spatial_probs], {
                self.state_input: state,
                self.mask_input: mask
            })
        return util.sample_action(self.interface, nonspatial_probs, spatial_probs), None

    def train_log_probs(self):
        return self._train_log_probs(self.nonspatial_probs, spatial_probs=self.spatial_probs)

    def train_values(self):
        return parts.value_head(self.features)


class LSTMAgent(InterfaceAgent):
    def __init__(self, interface):
        super().__init__(interface)
        self.num_select_actions = self.interface.num_select_unit_actions()

        self.rnn_size = 256
        self.self_attention = SelfAttention(hidden_size=128, num_heads=2, attention_dropout=0, train=True)
        self.lstm = tf.contrib.rnn.LSTMCell(self.rnn_size)

        self.lstm_memory_input = tf.placeholder(tf.float32, [2, None, self.rnn_size], name="memory_input")
        self.unit_embeddings_input = tf.placeholder(tf.float32, [None, None, self.interface.unit_embedding_size],
                                                    name="unit_embeddings_input") # [batch, num_units, embed_size]
        self.unit_selection_input = tf.placeholder(tf.int32, [None], name="unit_selection_input")
        self.prev_action_input = tf.placeholder(tf.int32, [None], name='prev_action_input')  # TODO: Implement this
        self.gltl_state = tf.placeholder(tf.int32, [None], name='gltl_state')

        self.features = self.features()  # Shape [batch_size, num_features]
        lstm_output, self.next_lstm_state = util.lstm_step(self.lstm, self.features, self.lstm_memory_input)

        self.state_values = parts.value_head(lstm_output)
        self.nonspatial_probs, self.spatial_probs = self._probs_from_features(lstm_output)
        self.unit_selection_probs = self._selection_probs_from_features(lstm_output, self.unit_embeddings_input)

    def get_initial_memory(self, num_agents):
        return [(np.zeros((2, self.rnn_size)), 0)] * num_agents

    def _selection_probs_from_features(self, features, embeddings):
        num_steps = tf.shape(self.mask_input)[0]
        return parts.actor_pointer_head(features[:num_steps], embeddings[:num_steps], self.num_select_actions)

    def features(self):
        conv_features = parts.conv_body(self.state_input)
        unit_features = tf.reduce_sum(self.self_attention(self.unit_embeddings_input, bias=0), axis=1)
        gltl_features = tf.one_hot(self.gltl_state, 4)  # self.interface.num_gltl_states)
        return tf.concat([conv_features, unit_features, gltl_features], axis=1)

    def next_mental_state(self, state, mental_state):
        return 1

    def step(self, states, masks, memory):
        """
        :param states: List of states of length batch size. In this case, state is a dict with keys:
            "unit_embeddings": numpy array with shape [num_units, embedding_size]
            "state": numpy array with shape [*state_shape]
        :param masks: numpy array of shape [batch_size, num_actions]
        :param memory: list of length size of tuples:
            [([2, memory_size], []), ...] or None for the first step
        """

        feed_dict = self.get_feed_dict(states, memory, masks)
        next_lstm_state, nonspacial_probs, spatial_probs, selection_probs = self.session.run(
            [self.next_lstm_state, self.nonspatial_probs, self.spatial_probs, self.unit_selection_probs], feed_dict)
        new_memory = [(next_lstm_state[:, i, :], self.next_mental_state(states[i], memory[i][1]))
                      for i in range(len(states))]
        unit_coords = util.pad_stack([state['unit_coords'][:, :2] for state in states], pad_axis=0, stack_axis=0)
        return util.sample_action(self.interface, nonspacial_probs, spatial_probs, selection_probs, unit_coords), new_memory

    def get_feed_dict(self, states, memories, masks, actions=None):
        """
        :param states:
        :param memories: List of length batch size containing
        :param masks:
        :param actions:
        :return:
        """
        screens = np.stack([state['screen'] for state in states], axis=0)
        lstm_memory = np.stack([memory[0] for memory in memories], axis=1)
        gltl_memory = np.stack([memory[1] for memory in memories], axis=0)

        feed_dict = {
            self.state_input: np.array(screens),  # TODO: Check why passing in states works for this:
            self.mask_input: np.array(masks),
            self.lstm_memory_input: lstm_memory,
            self.gltl_state: np.array(gltl_memory),
        }
        unit_embeddings = util.pad_stack([state['unit_embeddings'] for state in states], pad_axis=0, stack_axis=0)
        feed_dict[self.unit_embeddings_input] = unit_embeddings

        if actions is not None:
            nonspacial, spacials, selection_coords, selection_indices = zip(*[a.as_tuple() for a in actions])
            spacials = [(13, 27) if spacial is None else spacial for spacial in spacials]
            selections = [-1 if selection is None else selection for selection in selection_indices]
            feed_dict[self.action_input] = np.array(nonspacial)
            feed_dict[self.spacial_input] = np.array(spacials)
            feed_dict[self.unit_selection_input] = np.array(selections)
        return feed_dict

    def train_values(self):
        return self.state_values

    def train_log_probs(self):
        return self._train_log_probs(self.nonspatial_probs, self.spatial_probs, self.unit_selection_probs)
