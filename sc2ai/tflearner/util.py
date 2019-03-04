import numpy as np
import tensorflow as tf
import sys


class Util:
    @staticmethod
    def index(source, indices):
        """
        :param source: shape [T, num_actions, (N)] tensor
        :param indices: shape [T] tensor, each element is an integer in [0, num_actions)
        :return:
            A tensor of shape [T, (N)], each element is an element in the corresponding row of source, selected by
            the corresponding index in indices
        """
        num_choices = tf.shape(source)[1]
        oh = tf.one_hot(indices, num_choices)

        n = tf.size(tf.shape(source)) - 2
        extra_ones = tf.ones((n,), dtype=tf.int32)
        oh = tf.reshape(oh, tf.concat([tf.shape(source)[:2], extra_ones], axis=0))
        mult = oh * source
        # print_op = tf.print("tensors:", tf.shape(indices), tf.shape(oh), tf.shape(source),
        #                     output_stream=sys.stdout)
        # with tf.control_dependencies([print_op]):
        result = tf.reduce_sum(mult, axis=-1)
        return result

    @staticmethod
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
