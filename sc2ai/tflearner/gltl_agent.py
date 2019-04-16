import tensorflow as tf
import numpy as np

from sc2ai.tflearner import InterfaceAgent, SelfAttention
from sc2ai.tflearner import parts
from sc2ai.tflearner import util
from sc2ai.env_interface import ParamType


class GLTLLSTMAgent(InterfaceAgent):
    def __init__(self, interface):
        super().__init__(interface)
        self.num_select_actions = self.interface.num_unit_selection_actions

        self.rnn_size = 256
        self.self_attention = SelfAttention(hidden_size=128, num_heads=2, attention_dropout=0, train=True)
        self.lstm = tf.contrib.rnn.LSTMCell(self.rnn_size)

        self.lstm_memory_input = tf.placeholder(tf.float32, [2, None, self.rnn_size], name="memory_input")
        self.unit_embeddings_input = tf.placeholder(tf.float32, [None, None, self.interface.unit_embedding_size],
                                                    name="unit_embeddings_input")
        self.unit_selection_input = tf.placeholder(tf.int32, [None], name="unit_selection_input")
        self.prev_action_input = tf.placeholder(tf.int32, [None], name='prev_action_input')  # TODO: Implement this
        self.gltl_state = tf.placeholder(tf.int32, [None], name='gltl_state')

        self.features = self.features()  # Shape [batch_size, num_features]
        lstm_output, self.next_lstm_state = self._lstm_step()

        self.all_values = parts.value_head(lstm_output)
        self.nonspacial_probs, self.spacial_probs_x, self.spacial_probs_y = self._probs_from_features(lstm_output)
        self.nonspacial_train, self.spacial_train_x, self.spacial_train_y = self._probs_from_features(lstm_output[:-1])

        self.unit_selection_probs = self._selection_probs_from_features(lstm_output, self.unit_embeddings_input)
        self.unit_selection_probs_train = self._selection_probs_from_features(lstm_output[:-1],
                                                                              self.unit_embeddings_input[:-1])

    def _selection_probs_from_features(self, features, embeddings):
        return parts.actor_pointer_head(features, embeddings, self.num_select_actions)

    def features(self):
        conv_features = parts.conv_body(self.state_input)
        unit_features = tf.reduce_sum(self.self_attention(self.unit_embeddings_input, bias=0), axis=1)
        gltl_features = tf.one_hot(self.gltl_state, self.interface.num_gltl_states)
        return tf.concat([conv_features, unit_features, gltl_features], axis=1)

    def next_mental_state(self, state, mental_state):
        return 1

    def step(self, states, masks, memory):
        """
        :param states: List of states of length batch size. In this case, state is a dict with keys:
            "unit_embeddings": numpy array with shape [num_units, embedding_size]
            "state": numpy array with shape [*state_shape]
        :param masks: numpy array of shape [batch_size, num_actions]
        # :param memory: list of length size of tuples:
            [([2, memory_size], []), ...] or None for the first step
        """
        if memory is None:
            memory = [(np.zeros((2, self.rnn_size)), 0) for _ in range(len(states))]

        feed_dict = {
            **self.get_feed_dict(states, memory, masks),
        }
        results = self.session.run(
            [self.next_lstm_state, self.nonspacial_probs, self.unit_selection_probs,
             *self.spacial_probs_x, *self.spacial_probs_y], feed_dict)
        next_lstm_state, nonspacial_probs, selection_probs = results[:3]
        spacial_probs = results[3:]

        spacial_probs_x = spacial_probs[:self.num_screen_dims]
        spacial_probs_y = spacial_probs[self.num_screen_dims:]

        new_memory = []
        for i in range(len(states)):
            new_memory.append(((next_lstm_state[:, i, :]), self.next_mental_state(states[i], memory[i][1])))

        unit_coords = util.pad_stack([state['unit_coords'][:, :2] for state in states], pad_axis=0, stack_axis=0)
        return self.sample_action_index_with_units(nonspacial_probs, spacial_probs_x,
                                                   spacial_probs_y, selection_probs, unit_coords), new_memory

    def _lstm_step(self):
        state_tuple = tf.unstack(self.lstm_memory_input, axis=0)
        flattened_state = self.features
        lstm_output, next_state_tuple = self.lstm(flattened_state, state=state_tuple)
        next_state = tf.stack(next_state_tuple, axis=0)
        return lstm_output, next_state

    def get_feed_dict(self, states, memories, masks, actions=None, bootstrap_state=None):
        """
        :param states:
        :param memories: List of length batch size containing
        :param masks:
        :param actions:
        :param bootstrap_state:
        :return:
        """
        screens = np.stack([state['screen'] for state in states], axis=0)
        lstm_memory = np.stack([memory[0] for memory in memories], axis=1)
        gltl_memory = np.stack([memory[1] for memory in memories], axis=0)

        feed_dict = {
            self.state_input: np.array(states),
            self.mask_input: np.array(masks),
            self.lstm_memory_input: lstm_memory,
            self.gltl_state: np.array(gltl_memory),
        }
        all_states = states if bootstrap_state is None else [*states, bootstrap_state]
        unit_embeddings = util.pad_stack([state['unit_embeddings'] for state in all_states], pad_axis=0, stack_axis=0)
        feed_dict[self.unit_embeddings_input] = unit_embeddings

        if bootstrap_state is not None:
            bootstrap_screen = np.expand_dims(bootstrap_state['screen'], axis=0)
            feed_dict[self.state_input] = np.concatenate([screens, bootstrap_screen], axis=0)
        else:
            feed_dict[self.state_input] = screens

        if actions is not None:
            nonspacial, spacials, selection_coords, selection_indices = zip(*actions)
            spacials = [(13, 27) if spacial is None else spacial for spacial in spacials]
            selections = [-1 if selection is None else selection for selection in selection_indices]
            feed_dict[self.action_input] = np.array(nonspacial)
            feed_dict[self.spacial_input] = np.array(spacials)
            feed_dict[self.unit_selection_input] = np.array(selections)
        return feed_dict

    def bootstrap_value(self):
        return self.all_values[-1]

    def train_values(self):
        return self.all_values[:-1]

    def train_log_probs(self):
        return self._train_log_probs_with_units(self.nonspacial_train, self.spacial_train_x, self.spacial_train_y,
                                                self.unit_selection_probs_train)

    def sample_action_index_with_units(self, nonspacial_probs, spacial_probs_x, spacial_probs_y,
                                       unit_distribution, unit_coords):
        """
        unit_distribution is an array of shape [num_games, num_select_actions, num_units]
        :return: Generates an list of action index tuple of type (nonspacial_index, (spacial_x, spacial_y))
        """
        actions = self.sample_action_index(nonspacial_probs, spacial_probs_x, spacial_probs_y)
        num_units = unit_distribution.shape[2]
        new_actions = []
        unit_coords = np.stack(unit_coords)

        for i, action in enumerate(actions):
            nonspacial_index, coords = action
            param_type, param_index = self.interface.action_parameter_type(nonspacial_index)
            if param_type is ParamType.SELECT_UNIT:
                unit_choice = np.random.choice(num_units, p=unit_distribution[i, param_index])
                new_actions.append((nonspacial_index, coords,
                                    unit_coords[i, unit_choice].astype(np.int32), unit_choice))
            else:
                new_actions.append((nonspacial_index, coords, None, None))
        return new_actions

    def _get_chosen_selection_probs(self, selection_probs, selection_choice):
        """
        :param selection_probs: Tensor of integers of shape [T, num_units, num_selection_actions]
        :param selection_choice: Tensor of shape [T] of type int
        :return:
        """
        selection_probs = util.index(selection_probs, selection_choice)  # [T, num_selection_actions]
        num_selection_actions = self.interface.num_unit_selection_actions

        index = (self.action_input - self.num_screen_dims) % tf.convert_to_tensor(num_selection_actions)
        return util.index(selection_probs, index)  # [T]

    def _train_log_probs_with_units(self, nonspacial_probs, spacial_probs_x, spacial_probs_y, selection_probs):
        nonspacial_log_probs = tf.log(util.index(nonspacial_probs, self.action_input) + 1e-10)

        # TODO: This only works if all screen dimensions are the same. Should pad to greatest length
        probs_y = self._get_chosen_spacial_prob(spacial_probs_y, self.spacial_input[:, 1])
        probs_x = self._get_chosen_spacial_prob(spacial_probs_x, self.spacial_input[:, 0])

        spacial_log_probs = tf.log(probs_x + 1e-10) + tf.log(probs_y + 1e-10)

        result = nonspacial_log_probs
        result = result + tf.where(self.action_input < self.num_screen_dims,
                                   x=spacial_log_probs,
                                   y=tf.zeros_like(spacial_log_probs))
        if self.num_select_actions == 0:
            return result

        probs_selection = self._get_chosen_selection_probs(selection_probs, self.unit_selection_input)
        selection_log_prob = tf.log(probs_selection + 1e-10)
        is_select_action = tf.logical_and(self.action_input >= self.num_screen_dims,
                                          self.action_input < self.num_screen_dims + self.num_select_actions)
        result = result + tf.where(is_select_action,
                                   x=selection_log_prob,
                                   y=tf.zeros_like(selection_log_prob))
        return result
