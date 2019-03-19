from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

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
                A list of indices that represents the chosen action at the state. This can be of any data type
                and will be passed back into self.get_feed_dict during training.
            memory:
                An arbitrary object that will be passed into step at the next timestep.
        """
        pass

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


class InterfaceAgent(ActorCriticAgent, ABC):
    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.num_actions = self.interface.num_actions
        self.num_screen_dims = int(len(self.interface.screen_dimensions) / 2)

        self.state_input = tf.placeholder(tf.float32, [None, *self.interface.state_shape])  # [batch, *state_shape]
        self.mask_input = tf.placeholder(tf.float32, [None, self.interface.num_actions])  # [batch, num_actions]
        self.action_input = tf.placeholder(tf.int32, [None])   # [T]
        self.spacial_input = tf.placeholder(tf.int32, [None, 2])  # [T, 2]   dimension size 2 for x and y

    def get_feed_dict(self, states, masks, actions=None, bootstrap_state=None):
        feed_dict = {
            self.state_input: np.array(states),
            self.mask_input: np.array(masks),
        }
        if actions is not None:
            nonspacial, spacial = zip(*actions)
            spacial = [(13, 27) if spacial is None else spacial for spacial in spacial]
            feed_dict[self.action_input] = np.array(nonspacial)
            feed_dict[self.spacial_input] = np.array(spacial)
        return feed_dict

    def sample_action_index(self, nonspacial_probs, spacial_probs_x, spacial_probs_y):
        chosen_nonspacials = util.sample_multiple(nonspacial_probs)  # [num_games]
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

    def _train_log_probs(self, nonspacial_probs, spacial_probs_x, spacial_probs_y):
        nonspacial_log_probs = tf.log(util.index(nonspacial_probs, self.action_input) + 0.00000001)

        # TODO: This only works if all screen dimensions are the same. Should pad to greatest length
        probs_y = self._get_chosen_spacial_prob(spacial_probs_y, self.spacial_input[:, 1])
        probs_x = self._get_chosen_spacial_prob(spacial_probs_x, self.spacial_input[:, 0])
        spacial_log_probs = tf.log(probs_x + 0.0000001) + tf.log(probs_y + 0.0000001)
        result = nonspacial_log_probs + tf.where(self.action_input < self.num_screen_dims,
                                                 x=spacial_log_probs,
                                                 y=tf.zeros_like(spacial_log_probs))
        return result

    def _get_chosen_spacial_prob(self, spacial_probs, spacial_choice):
        spacial_probs = tf.stack(spacial_probs, axis=-1)  # [T, screen_dim, num_screen_dimensions]
        spacial_probs = util.index(spacial_probs, spacial_choice)  # [T, num_screen_dimensions]
        return util.index(spacial_probs, self.action_input % tf.convert_to_tensor(self.num_screen_dims))  # [T]

    def _probs_from_features(self, features):
        nonspacial_probs = parts.actor_nonspacial_head(features, self.mask_input, self.num_actions)
        spacial_probs_x, spacial_probs_y = \
            parts.actor_spacial_head(features, self.interface.screen_dimensions)
        return nonspacial_probs, spacial_probs_x, spacial_probs_y


class ConvAgent(InterfaceAgent):
    def __init__(self, interface):
        super().__init__(interface)
        self.bootstrap_state_input = tf.placeholder(tf.float32, [*self.interface.state_shape])

        self.features = parts.conv_body(self.state_input)
        self.nonspacial_probs, self.spacial_probs_x, self.spacial_probs_y = self._probs_from_features(self.features)

    def step(self, state, mask, memory):
        probs = self.session.run(
            [self.nonspacial_probs, *self.spacial_probs_x, *self.spacial_probs_y], {
                self.state_input: state,
                self.mask_input: mask
            })
        nonspacial_probs = probs[0]
        spacial_probs = probs[1:]
        spacial_probs_x = spacial_probs[:self.num_screen_dims]
        spacial_probs_y = spacial_probs[self.num_screen_dims:]  # [num_screen_dims, num_games, screen_dim]

        return self.sample_action_index(nonspacial_probs, spacial_probs_x, spacial_probs_y), None

    def get_feed_dict(self, states, masks, actions=None, bootstrap_state=None):
        feed_dict = super(ConvAgent, self).get_feed_dict(states, masks, actions, bootstrap_state)
        return {
            self.bootstrap_state_input: np.array(bootstrap_state),
            **feed_dict
        }

    def train_log_probs(self):
        return self._train_log_probs(self.nonspacial_probs, self.spacial_probs_x, self.spacial_probs_y)

    def bootstrap_value(self):
        return tf.squeeze(parts.value_head(parts.conv_body(tf.expand_dims(self.bootstrap_state_input, axis=0))), axis=0)

    def train_values(self):
        return parts.value_head(self.features)


class LSTMAgent(InterfaceAgent):
    def __init__(self, interface):
        super().__init__(interface)

        self.rnn_size = 256
        self.self_attention = SelfAttention(hidden_size=128, num_heads=2, attention_dropout=0, train=True)
        self.lstm = tf.contrib.rnn.LSTMCell(self.rnn_size)

        self.memory_input = tf.placeholder(tf.float32, [2, None, self.rnn_size])
        self.unit_embeddings_input = tf.placeholder(tf.float32, [None, None, self.interface.unit_embedding_size])

        self.features = self.features()

        lstm_output, self.next_state = self._lstm_step()
        self.train_output = self._lstm_step_train()
        self.all_values = parts.value_head(self.train_output)

        self.nonspacial_probs, self.spacial_probs_x, self.spacial_probs_y = self._probs_from_features(lstm_output)
        self.nonspacial_train, self.spacial_train_x, self.spacial_train_y = \
            self._probs_from_features(self.train_output[:-1])

    def features(self):
        conv_features = parts.conv_body(self.state_input)
        unit_features = tf.reduce_sum(self.self_attention(self.unit_embeddings_input, bias=0), axis=1)
        return tf.concat([conv_features, unit_features], axis=1)

    def step(self, states, masks, memory):
        """
        :param states: List of states of length batch size. Usually numpy arrays of shape [*state_size]
        :param masks: numpy array of shape [batch_size, num_actions]
        :param memory: numpy of shape [2, batch_size, memory_size] or None for the first step
        """
        if memory is None:
            memory = np.zeros((2, len(states), self.rnn_size))

        feed_dict = {
            **self.get_feed_dict(states, masks),
            self.memory_input: memory
        }
        results = self.session.run(
            [self.next_state, self.nonspacial_probs, *self.spacial_probs_x, *self.spacial_probs_y], feed_dict)
        next_state = results[0]
        nonspacial_probs = results[1]
        spacial_probs = results[2:]
        spacial_probs_x = spacial_probs[:self.num_screen_dims]
        spacial_probs_y = spacial_probs[self.num_screen_dims:]

        return self.sample_action_index(nonspacial_probs, spacial_probs_x, spacial_probs_y), next_state

    def _lstm_step(self):
        state_tuple = tf.unstack(self.memory_input, axis=0)
        flattened_state = self.features
        lstm_output, next_state_tuple = self.lstm(flattened_state, state=state_tuple)
        next_state = tf.stack(next_state_tuple, axis=0)
        return lstm_output, next_state

    def _lstm_step_train(self):
        flattened_state = tf.expand_dims(self.features, axis=0)
        train_output, _ = tf.nn.dynamic_rnn(self.lstm, flattened_state, dtype=tf.float32)
        return tf.squeeze(train_output, axis=0)

    def get_feed_dict(self, states, masks, actions=None, bootstrap_state=None):
        screens = np.stack([state['screen'] for state in states], axis=0)
        unit_embeddings = np.stack([state['unit_embeddings'] for state in states], axis=0)

        feed_dict = super(LSTMAgent, self).get_feed_dict(screens, masks, actions)
        if bootstrap_state is not None:
            bootstrap_screen = np.expand_dims(bootstrap_state['screen'], axis=0)
            bootstrap_units = np.expand_dims(bootstrap_state['unit_embeddings'], axis=0)
            feed_dict[self.state_input] = np.concatenate([screens, bootstrap_screen], axis=0)
            feed_dict[self.unit_embeddings_input] = np.concatenate([unit_embeddings, bootstrap_units], axis=0)
        else:
            feed_dict[self.state_input] = screens
            feed_dict[self.unit_embeddings_input] = unit_embeddings
        return feed_dict

    def train_log_probs(self):
        return self._train_log_probs(self.nonspacial_train, self.spacial_train_x, self.spacial_train_y)

    def bootstrap_value(self):
        return self.all_values[-1]

    def train_values(self):
        return self.all_values[:-1]





















