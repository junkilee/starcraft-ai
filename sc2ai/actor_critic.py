from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class ActorCritic(ABC):
    def __init__(self, num_actions, state_shape):
        self.num_actions = num_actions
        self.state_shape = state_shape

    @abstractmethod
    def actor(self, states):
        """
        Tensorflow operation that maps states to actions
        :param states: A tensor of shape [num_instances, num_steps, *state_shape]
        :return: A tensor of shape [num_instances, num_steps, num_actions]
        """
        pass

    @abstractmethod
    def critic(self, states):
        """
        Tensorflow operation that maps states to values
        :param states: A tensor of shape [num_instances, num_steps, *state_shape]
        :return: A tensor of shape [num_instances, num_steps]
        """
        pass


class BasicActorCritic(ActorCritic):
    def __init__(self, num_actions, state_shape, architecture=(32, 32), shared_architecture=(),
                 dropout_prob=0):

        super().__init__(num_actions, state_shape)
        self.architecture = architecture
        self.shared_architecture = shared_architecture
        self.dropout_prob = dropout_prob

    def shared_layers(self, state):
        state = tf.reshape(state, [1, np.prod(self.state_shape)])
        with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.shared_architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='shared%d' % i)
        return state

    def actor(self, state):
        state = self.shared_layers(state)
        with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='layer%d' % i)
            probs = tf.layers.dense(state, units=self.num_actions, activation=tf.nn.softmax, name='output')
        return tf.squeeze(probs)

    def critic(self, state):
        state = self.shared_layers(state)
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.architecture):
                state = tf.layers.dense(state, layer_size, activation=tf.nn.tanh, name='critic%d' % i)
            state = tf.layers.dense(state, 1, activation=None, name='output')
        return tf.squeeze(state, axis=-1)
