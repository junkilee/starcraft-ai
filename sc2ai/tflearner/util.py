import numpy as np
import tensorflow as tf

from sc2ai.env_interface import AgentAction


def sample_action(num_spatial_actions, nonspatial_probs, spatial_probs):
    """
    :return: Generates a list of AgentActions
    """
    chosen_nonspatials = sample_multiple(nonspatial_probs)  # [num_games]
    agent_actions = []
    for i, chosen_nonspatial in enumerate(chosen_nonspatials):
        if chosen_nonspatial < num_spatial_actions:
            x = np.random.choice(84, p=spatial_probs[0, i, :, chosen_nonspatial])
            y = np.random.choice(84, p=spatial_probs[1, i, :, chosen_nonspatial])
            agent_actions.append(AgentAction(chosen_nonspatial, spatial_coords=(x, y)))
        else:
            agent_actions.append(AgentAction(chosen_nonspatial))
    return agent_actions


def pad_stack(arrays, pad_axis, stack_axis):
    """
    Similar to np.stack, but for arrays whose shapes differ in one dimension. First pads all arrays to match the size
    of the largest array in that dimension with 0. Then calls np.stack.

    :param arrays: List of numpy arrays. Arrays have must be the same shape except in the `pad_axis` axis.
    :param pad_axis: The axis to pad.
    :param stack_axis: The axis in the result array along which the input arrays are stacked.

    :return: The stacked array which has one more dimension than the input arrays.
    """
    max_dim_size = np.max([arr.shape[pad_axis] for arr in arrays])
    paddeds = []
    for array in arrays:
        shape = np.array(array.shape)
        shape[pad_axis] = max_dim_size - shape[pad_axis]
        padding = np.zeros(shape)
        padded = np.concatenate([array, padding], axis=pad_axis)
        paddeds.append(padded)
    return np.stack(paddeds, axis=stack_axis)


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
