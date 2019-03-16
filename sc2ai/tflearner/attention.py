"""
Implementation of Scaled Dot Product Attention
https://arxiv.org/abs/1706.03762

The original source code adapted from
https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py
"""
import numpy as np
import tensorflow as tf

class MultiHeadedAttention:
    """
    A Class for Multi Headed Attention Module

    :param num_heads(int): The Number of heads in the attention module.
    :param d_model(int): The dimension of the model.
    :param d_q(int): The dimension of a query
    :param d_k(int): The dimension of a key
    :param d_v(int): The dimension of a value
    :param dropout(float): The probability of dropout
    :param masked(bool): The probability of dropout
    """
    def __init__(self, num_heads, d_model, d_q, d_k, d_v, dropout, masked):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.masked = masked

    def _linear_projection(self, q, k, v):
        with tf.variable_scope('attention_linear_projection', reuse=tf.AUTO_REUSE):
            q = tf.layers.dense(q, self.d_q, use_bias=False)
            k = tf.layers.dense(k, self.d_k, use_bias=False)
            v = tf.layers.dense(v, self.d_V, use_bias=False)
        return q, k, v

    def _attention(self, qs, ks, vs):
        d_k_per_head = self.d_k // self.num_heads

        qk = tf.matmul(qs, ks, transpose_b=True)
        before_softmax = qk / (d_k_per_head ** 0.5)

        if self.masked:
            diag_vals = tf.ones_like(before_softmax[0, 0, :, :])  # (batch_size, num_heads, query_dim, key_dim)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (q_dim, k_dim)
            masks = tf.tile(tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
                            [tf.shape(before_softmax)[0], tf.shape(before_softmax)[1], 1, 1])
            paddings = tf.ones_like(masks) * -1e9
            before_softmax = tf.where(tf.equal(masks, 0), paddings, before_softmax)

        after_softmax = tf.nn.softmax(before_softmax)
        return tf.matmul(after_softmax, vs)

    def _split_heads(self, q, k, v):
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.d_q)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.d_k)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.d_v)

        return qs, ks, vs

    def _concat_heads(self, head_outputs):
        tensor = tf.transpose(head_outputs, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
        t_shape = tensor.get_shape().as_list()
        num_heads, dim = t_shape[-2:]
        return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

    def multihead(self, q, k, v): # q==k==v
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._attention(qs, ks, vs)
        output = self._concat_heads(outputs)
        with tf.variable_scope('attention_w_o', reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(output, self.d_model)

        return tf.nn.dropout(output, 1.0 - self.dropout)

