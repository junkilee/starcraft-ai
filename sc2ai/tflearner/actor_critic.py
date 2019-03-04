from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class ActorCritic(ABC):
    def __init__(self, num_actions, state_shape):
        self.num_actions = num_actions
        self.state_shape = state_shape

    @abstractmethod
    def actor(self, states, action_mask):
        """
        Tensorflow operation that maps states to actions
        :param states: A tensor of shape [num_steps, *state_shape]
        :param action_mask: A tensor of shape [num_steps, num_actions]
        :return: A tensor of shape [num_steps, num_actions]
        """
        pass

    @abstractmethod
    def critic(self, states):
        """
        Tensorflow operation that maps states to values
        :param states: A tensor of shape [num_steps, *state_shape]
        :return: A tensor of shape [num_steps, num_steps]
        """
        pass


class BasicActorCritic(ActorCritic):
    """
    Actor critic model that takes the entire state and flattens it, and performs feed
    forward layers on the flattened state.
    """
    def __init__(self, num_actions, state_shape, architecture=(32, 32), shared_architecture=(),
                 dropout_prob=0):

        super().__init__(num_actions, state_shape)
        self.architecture = architecture
        self.shared_architecture = shared_architecture
        self.dropout_prob = dropout_prob

    def shared_layers(self, state):
        state = tf.reshape(state, [-1, np.prod(self.state_shape)])
        with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.shared_architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='shared%d' % i)
        return state

    def actor(self, state, action_mask):
        state = self.shared_layers(state)
        with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='layer%d' % i)
            probs = tf.layers.dense(state, units=self.num_actions, activation=tf.nn.softmax, name='output')
        masked = probs * action_mask
        return masked / tf.reduce_sum(masked, axis=1, keepdims=True)

    def critic(self, state):
        state = self.shared_layers(state)
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='critic%d' % i)
            state = tf.layers.dense(state, 1, activation=None, name='output')
        return tf.squeeze(state, axis=-1)


# TODO Refactor the rest of the code above to match ConvActorCritic
class ConvActorCritic:
    """
    Assumes the state shape is 3 dimensional. Applies some number of convolution layers
    and feed forward layers.
    """
    def __init__(self, num_spacial_actions, spacial_dimensions, state_shape, dropout_prob=0):
        self.num_actions = num_spacial_actions
        self.spacial_dimensions = spacial_dimensions

        self.state_shape = state_shape
        self.reward_architecture = (32,)
        self.dropout_prob = dropout_prob

        self.filters = [16, 16, 16, 16]
        self.kernel_sizes = [7, 5, 3, 3]
        self.strides = [3, 2, 1, 1]
        self.architecture = list(zip(range(len(self.filters)), self.filters, self.kernel_sizes, self.strides))

    def shared_layers(self, state):
        logits = tf.transpose(state, [0, 3, 2, 1])  # Transpose into channels last format
        with tf.variable_scope('ac_shared', reuse=tf.AUTO_REUSE):
            for i, filters, kernel_size, strides in self.architecture:
                logits = tf.layers.conv2d(logits, filters, kernel_size, strides, activation=tf.nn.leaky_relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          name='conv%d' % i)
            logits = tf.layers.flatten(logits)
            logits = tf.layers.dense(logits, units=64, activation=tf.nn.leaky_relu, name='d1')

        logits = tf.reshape(logits, [-1, 64])
        return logits

    def actor_spacial(self, state, sizes):
        features = self.shared_layers(state)
        spacial_x = []
        spacial_y = []
        with tf.variable_scope('actor_spacial_x', reuse=tf.AUTO_REUSE):
            for i in range(int(len(sizes) / 2)):
                spacial_x.append(tf.layers.dense(features,
                                                 units=sizes[i * 2], activation=tf.nn.softmax, name='output%d' % i))
        with tf.variable_scope('actor_spacial_y', reuse=tf.AUTO_REUSE):
            for i in range(int(len(sizes) / 2)):
                spacial_y.append(tf.layers.dense(features,
                                                 units=sizes[i * 2 + 1], activation=tf.nn.softmax, name='output%d' % i))
        return spacial_x, spacial_y

    def actor_nonspacial(self, state, action_mask):
        features = self.shared_layers(state)
        with tf.variable_scope('actor_nonspacial', reuse=tf.AUTO_REUSE):
            probs = tf.layers.dense(features, units=self.num_actions, activation=tf.nn.softmax, name='output')
        masked = (probs + 0.000001) * action_mask
        return masked / tf.reduce_sum(masked, axis=1, keepdims=True)

    def critic(self, state):
        features = self.shared_layers(state)
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            features = tf.layers.dense(features, units=1, activation=None, name='output')
        return tf.squeeze(features, axis=-1)
