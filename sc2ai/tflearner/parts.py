import tensorflow as tf
import numpy as np

def conv_body_with_dense(state, filters=(16,), kernel_sizes=(3,), strides=(3,), output_size=64):
    """
    Applies some number of convolution layers, followed by a single dense layer.

    :param state: Tensor of shape [channels, batch, x_dim, y_dim]
    :param filters: Tuple of filters sizes.
    :param kernel_sizes: Tuple of kernel sizes.
    :param strides: Tuple of stride sizes. All tuples must be the same length.
    :param output_size: The number of units to output to.
    :return: A tensor of shape [batch_size, output_size]
    """
    tensor = conv_body(state, filters, kernel_sizes, strides)
    with tf.variable_scope('ac_shared', reuse=tf.AUTO_REUSE):
        logits = tf.layers.flatten(tensor)
        logits = tf.layers.dense(logits, units=output_size, activation=tf.nn.leaky_relu, name='dense1')

    logits = tf.reshape(logits, [-1, output_size])
    return logits

def conv_body(tensor, filters=(16,), kernel_sizes=(3,), strides=(3,), output_channels=4):
    """
    Applies some number of convolution layers

    :param tensor: Tensor of shape [batch, x_dim, y_dim, channels]
    :param filters: Tuple of filters sizes.
    :param kernel_sizes: Tuple of kernel sizes.
    :param strides: Tuple of stride sizes. All tuples must be the same length.
    :return: Tensor of shape [batch, x_dim, y_dim, channels]
    """
    architecture = list(zip(range(len(filters)), filters, kernel_sizes, strides))

    with tf.variable_scope('ac_shared', reuse=tf.AUTO_REUSE):
        for i, filters, kernel_size, strides in architecture:
            print("DEBUG: Convolutional Layer", i, "Filter size", filters, \
                "Kernel", kernel_size, "Stride", strides, "Current Tensor", tensor)
            tensor = tf.layers.conv2d(tensor, filters, kernel_size, strides, activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      name='conv{}'.format(i))
        # # 1x1 convolution to change output shape
        # tensor = tf.layers.conv2d(tensor, output_channels, 1, 1, activation=tf.nn.leaky_relu,
        #                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        #                               name='conv_combine')
    return tensor


def actor_spatial_head(features, screen_dim, num_spatial_actions, name='actor_spatial_x', from_conv=False):
    """
    Feed forward network to calculate the spacial action probabilities.

    :param name: Name of scope. Change name to have a different set of variables
    :param features: Tensor of shape [batch_size, num_features] inputs to the spacial_head
    :param screen_dim: Number of units per distribution, corresponds to width / height of screen.
    :param num_spatial_actions: Number of distributions over spatial coordinates for each x, y to produce.

    :return:
        Tensor of shape [2, batch_size, screen_dim, num_spatial_actions], Last Convolution before flatten
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if from_conv:
            # 1x1 convolution to change output shape, 1 for each spatial action         
            features = tf.layers.conv2d(features, num_spatial_actions, 1, 1, activation=None,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      name='conv_combine')
            # resize to the correct size for output
            shape = features.get_shape().as_list()
            size_diff = shape[1] * shape[2]/(screen_dim*screen_dim)
            print("DEBUG: size diff", size_diff)
            probs_2d = tf.image.resize_bilinear(features, (screen_dim, screen_dim), name='resize_up')

            # Softmax each 84x84 channel by (reshape [-1,84x84,ch] -> softmax -> reshape [-1,84,84,ch])
            probs_flat = tf.reshape(probs_2d, [-1, screen_dim * screen_dim, num_spatial_actions])
            softmax_temp = 0.001 # size_diff * 1 # Temperature - Use your actions early!
            print("DEBUG: Temperature", softmax_temp)
            softmax_flat = tf.nn.softmax(probs_flat / softmax_temp, axis=1)
            softmax_probs_2d = tf.reshape(softmax_flat, [-1, screen_dim, screen_dim, num_spatial_actions])
            
            return softmax_probs_2d
            # # Each action got to choose what is important in the convolution above.
            # # Now we sum across x to get y probs and sum across y to get x probs.
            # y_flattened = tf.reduce_sum(probs_2d, axis=1)
            # x_flattened = tf.reduce_sum(probs_2d, axis=2)
            # # Reshape to the correct format: [2, batch, 84, actions]
            # xy_distribution = tf.stack([y_flattened, x_flattened], axis=0)
        else:
            print("DEBUG: function deprecated without 'from_conv=True' ")
            return None

    # return tf.nn.softmax(xy_distribution, axis=-2), softmax_probs_2d


def actor_pointer_head(features, embeddings, num_heads, name='pointer_head'):
    """
    Feed forward network that performs attention on on the embeddings using features `num_head` times.

    :param name: Name of scope. Change name to have a different set of variables
    :param features: Tensor of shape `[batch_size, num_features]`
    :param embeddings: Tensor of shape `[batch_size, num_units, embedding_size]`
    :param num_heads: An integer representing the number of separate softmax distributions to output.

    :return: A softmax distribution over the units in the embedding of shape `[batch_size, num_heads, num_units]`
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        hidden_size = 50
        mapped_features = tf.layers.dense(features, hidden_size, activation=None, use_bias=False)
        mapped_embeddings = tf.layers.dense(embeddings, hidden_size, activation=None, use_bias=False)

        mapped_features = tf.expand_dims(mapped_features, axis=1)
        logits = tf.layers.dense(tf.tanh(mapped_features + mapped_embeddings), num_heads,
                                 activation=None, use_bias=False)
        logits = tf.transpose(logits, [0, 2, 1])  # Now shape [batch_size, num_heads, num_units]

        # If embedding is all zeros, then this "unit" is padding and should be masked out
        is_real_unit_mask = tf.not_equal(tf.reduce_sum(embeddings, axis=-1), tf.constant(0, dtype=tf.float32))
        is_real_unit_mask = tf.expand_dims(is_real_unit_mask, axis=1)

        # TODO: Check math to make sure this is equivalent to not masking:
        probs = tf.nn.softmax(logits + 1e-10) * tf.cast(is_real_unit_mask, dtype=tf.float32) + 1e-10
        return probs / tf.reduce_sum(probs, axis=-1, keepdims=True)


def actor_nonspatial_head(features, action_mask, num_actions, name='actor_nonspatial', from_conv=False):
    """
    Feed forward network to produce the nonspatial action probabilities.

    :param features: Tensor of shape [batch_size, num_features]
    :param action_mask: Tensor of shape [batch_size, num_actions]
    :param num_actions: number of actions to produce.
    :param name: Name of scope. Change name to have a different set of variables
    :return: Tensor of shape [batch_size, num_actions] of probability distributions.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if from_conv:
            features = tf.layers.conv2d(features, num_actions, 1, 1, activation=None,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          name='conv_combine')

            features =  tf.layers.flatten(features)
        probs = tf.layers.dense(features, units=num_actions, activation=tf.nn.softmax, name='output')
    masked = (probs + 1e-10) * action_mask
    return masked / tf.reduce_sum(masked, axis=1, keepdims=True)

def value_head(features, name='critic', from_conv=False):
    """
    Feed forward network to produce the state value.

    :param name: Name of scope. Change name to have a different set of variables
    :param features: Tensor of shape [batch_size, num_features].
    :return: Tensor of shape [batch_size] of state values.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if from_conv:
            features =  tf.layers.flatten(features)
        features = tf.layers.dense(features, units=1, activation=None, name='output')
    return tf.squeeze(features, axis=-1)
