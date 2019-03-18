import numpy as np
import tensorflow as tf


def index(source, indices):
    """
    :param source: shape [T, num_actions, (N)] tensor. (N) is any number of additional dimensions.
    :param indices: shape [T] tensor, each element is an integer in [0, num_actions)
    :return:
        A tensor of shape [T, (N)], each element is an element in the corresponding row of source, selected by
        the corresponding index in indices
    """
    num_choices = tf.shape(source)[1]
    one_hot = tf.one_hot(indices, num_choices)

    n = tf.size(tf.shape(source)) - 2
    extra_ones = tf.ones((n,), dtype=tf.int32)
    broadcast_shape = tf.concat([tf.shape(source)[:2], extra_ones], axis=0)
    return tf.reduce_sum(tf.reshape(one_hot, broadcast_shape) * source, axis=1)


def sample_multiple(probabilities):
    """
    Treats each row of a 2d matrix as a probability distribution and returns sampled indices
    :param probabilities: predictions with shape [batch_size, output_size]
    :return: sampled indices with shape [batch_size]
    """
    cumulative = probabilities.cumsum(axis=1)
    uniform = np.random.rand(len(cumulative), 1)
    choices = (uniform < cumulative).argmax(axis=1)
    return choices
