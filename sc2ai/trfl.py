import collections

# Dependency imports
import tensorflow as tf

def generalized_lambda_returns(rewards,
                               pcontinues,
                               values,
                               bootstrap_value,
                               lambda_=1,
                               name="generalized_lambda_returns"):
  """Computes lambda-returns along a batch of (chunks of) trajectories.

  For lambda=1 these will be multistep returns looking ahead from each
  state to the end of the chunk, where bootstrap_value is used. If you pass an
  entire trajectory and zeros for bootstrap_value, this is just the Monte-Carlo
  return / TD(1) target.

  For lambda=0 these are one-step TD(0) targets.

  For inbetween values of lambda these are lambda-returns / TD(lambda) targets,
  except that traces are always cut off at the end of the chunk, since we can't
  see returns beyond then. If you pass an entire trajectory with zeros for
  bootstrap_value though, then they're plain TD(lambda) targets.

  lambda can also be a tensor of values in [0, 1], determining the mix of
  bootstrapping vs further accumulation of multistep returns at each timestep.
  This can be used to implement Retrace and other algorithms. See
  `sequence_ops.multistep_forward_view` for more info on this. Another way to
  think about the end-of-chunk cutoff is that lambda is always effectively zero
  on the timestep after the end of the chunk, since at the end of the chunk we
  rely entirely on bootstrapping and can't accumulate returns looking further
  into the future.

  The sequences in the tensors should be aligned such that an agent in a state
  with value `V` transitions into another state with value `V'`, receiving
  reward `r` and pcontinue `p`. Then `V`, `r` and `p` are all at the same index
  `i` in the corresponding tensors. `V'` is at index `i+1`, or in the
  `bootstrap_value` tensor if `i == T`.

  Subtracting `values` from these lambda-returns will yield estimates of the
  advantage function which can be used for both the policy gradient loss and
  the baseline value function loss in A3C / GAE.

  Args:
    rewards: 2-D Tensor with shape `[T, B]`.
    pcontinues: 2-D Tensor with shape `[T, B]`.
    values: 2-D Tensor containing estimates of the state values for timesteps
      0 to `T-1`. Shape `[T, B]`.
    bootstrap_value: 1-D Tensor containing an estimate of the value of the
      final state at time `T`, used for bootstrapping the target n-step
      returns. Shape `[B]`.
    lambda_: an optional scalar or 2-D Tensor with shape `[T, B]`.
    name: Customises the name_scope for this op.

  Returns:
    2-D Tensor with shape `[T, B]`
  """
  values.get_shape().assert_has_rank(2)
  rewards.get_shape().assert_has_rank(2)
  pcontinues.get_shape().assert_has_rank(2)
  bootstrap_value.get_shape().assert_has_rank(1)
  scoped_values = [rewards, pcontinues, values, bootstrap_value, lambda_]
  with tf.name_scope(name, values=scoped_values):
    if lambda_ == 1:
      # This is actually equivalent to the branch below, just an optimisation
      # to avoid unnecessary work in this case:
      return scan_discounted_sum(
          rewards,
          pcontinues,
          initial_value=bootstrap_value,
          reverse=True,
          back_prop=False,
          name="multistep_returns")
    else:
      v_tp1 = tf.concat(
          axis=0, values=[values[1:, :],
                          tf.expand_dims(bootstrap_value, 0)])
      # `back_prop=False` prevents gradients flowing into values and
      # bootstrap_value, which is what you want when using the bootstrapped
      # lambda-returns in an update as targets for values.
      return multistep_forward_view(
          rewards,
          pcontinues,
          v_tp1,
          lambda_,
          back_prop=False,
          name="generalized_lambda_returns")




def multistep_forward_view(rewards, pcontinues, state_values, lambda_,
                           back_prop=True, sequence_lengths=None,
                           name="multistep_forward_view_op"):
  """Evaluates complex backups (forward view of eligibility traces).

    ```python
    result[t] = rewards[t] +
        pcontinues[t]*(lambda_[t]*result[t+1] + (1-lambda_[t])*state_values[t])
    result[last] = rewards[last] + pcontinues[last]*state_values[last]
    ```

    This operation evaluates multistep returns where lambda_ parameter controls
    mixing between full returns and boostrapping. It is users responsibility
    to provide state_values. Depending on how state_values are evaluated this
    function can evaluate targets for Q(lambda), Sarsa(lambda) or some other
    multistep boostrapping algorithm.

    More information about a forward view is given here:
      http://incompleteideas.net/sutton/book/ebook/node74.html

    Please note that instead of evaluating traces and then explicitly summing
    them we instead evaluate mixed returns in the reverse temporal order
    by using the recurrent relationship given above.

    The parameter lambda_ can either be a constant value (e.g for Peng's
    Q(lambda) and Sarsa(_lambda)) or alternatively it can be a tensor containing
    arbitrary values (Watkins' Q(lambda), Munos' Retrace, etc).

    The result of evaluating this recurrence relation is a weighted sum of
    n-step returns, as depicted in the diagram below. One strategy to prove this
    equivalence notes that many of the terms in adjacent n-step returns
    "telescope", or cancel out, when the returns are summed.

    Below L3 is lambda at time step 3 (important: this diagram is 1-indexed, not
    0-indexed like Python). If lambda is scalar then L1=L2=...=Ln.
    g1,...,gn are discounts.

    ```
    Weights:  (1-L1)        (1-L2)*l1      (1-L3)*l1*l2  ...  L1*L2*...*L{n-1}
    Returns:    |r1*(g1)+     |r1*(g1)+      |r1*(g1)+          |r1*(g1)+
              v1*(g1)         |r2*(g1*g2)+   |r2*(g1*g2)+       |r2*(g1*g2)+
                            v2*(g1*g2)       |r3*(g1*g2*g3)+    |r3*(g1*g2*g3)+
                                           v3*(g1*g2*g3)               ...
                                                                |rn*(g1*...*gn)+
                                                              vn*(g1*...*gn)
    ```

  Args:
    rewards: Tensor of shape `[T, B]` containing rewards.
    pcontinues: Tensor of shape `[T, B]` containing discounts.
    state_values: Tensor of shape `[T, B]` containing state values.
    lambda_: Mixing parameter lambda.
        The parameter can either be a scalar or a Tensor of shape `[T, B]`
        if mixing is a function of state.
    back_prop: Whether to backpropagate.
    sequence_lengths: Tensor of shape `[B]` containing sequence lengths to be
      (reversed and then) summed, same as in `scan_discounted_sum`.
    name: Sets the name_scope for this op.

  Returns:
      Tensor of shape `[T, B]` containing multistep returns.
  """
  with tf.name_scope(name, values=[rewards, pcontinues, state_values]):
    # Regroup:
    #   result[t] = (rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]) +
    #               pcontinues[t]*lambda_*result[t + 1]
    # Define:
    #   sequence[t] = rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]
    #   discount[t] = pcontinues[t]*lambda_
    # Substitute:
    #   result[t] = sequence[t] + discount[t]*result[t + 1]
    # Boundary condition:
    #   result[last] = rewards[last] + pcontinues[last]*state_values[last]
    # Add and subtract the same quantity at BC:
    #   state_values[last] =
    #       lambda_*state_values[last] + (1-lambda_)*state_values[last]
    # This makes:
    #   result[last] =
    #       (rewards[last] + pcontinues[last]*(1-lambda_)*state_values[last]) +
    #       pcontinues[last]*lambda_*state_values[last]
    # Substitute in definitions for sequence and discount:
    #   result[last] = sequence[last] + discount[last]*state_values[last]
    # Define:
    #   initial_value=state_values[last]
    # We get the following recurrent relationship:
    #   result[last] = sequence[last] + decay[last]*initial_value
    #   result[k] = sequence[k] + decay[k] * result[k + 1]
    # This matches the form of scan_discounted_sum:
    #   result = scan_sum_with_discount(sequence, discount,
    #                                   initial_value = state_values[last])
    sequence = rewards + pcontinues * state_values * (1 - lambda_)
    discount = pcontinues * lambda_
    return scan_discounted_sum(sequence, discount, state_values[-1],
                               reverse=True, sequence_lengths=sequence_lengths,
                               back_prop=back_prop)


def scan_discounted_sum(sequence, decay, initial_value, reverse=False,
                        sequence_lengths=None, back_prop=True,
                        name="scan_discounted_sum"):
  """Evaluates a cumulative discounted sum along dimension 0.

    ```python
    if reverse = False:
      result[1] = sequence[1] + decay[1] * initial_value
      result[k] = sequence[k] + decay[k] * result[k - 1]
    if reverse = True:
      result[last] = sequence[last] + decay[last] * initial_value
      result[k] = sequence[k] + decay[k] * result[k + 1]
    ```

  Respective dimensions T, B and ... have to be the same for all input tensors.
  T: temporal dimension of the sequence; B: batch dimension of the sequence.

    if sequence_lengths is set then x1 and x2 below are equivalent:
    ```python
    x1 = zero_pad_to_length(
      scan_discounted_sum(
          sequence[:length], decays[:length], **kwargs), length=T)
    x2 = scan_discounted_sum(sequence, decays,
                             sequence_lengths=[length], **kwargs)
    ```

  Args:
    sequence: Tensor of shape `[T, B, ...]` containing values to be summed.
    decay: Tensor of shape `[T, B, ...]` containing decays/discounts.
    initial_value: Tensor of shape `[B, ...]` containing initial value.
    reverse: Whether to process the sum in a reverse order.
    sequence_lengths: Tensor of shape `[B]` containing sequence lengths to be
      (reversed and then) summed.
    back_prop: Whether to backpropagate.
    name: Sets the name_scope for this op.

  Returns:
    Cumulative sum with discount. Same shape and type as `sequence`.
  """
  # Note this can be implemented in terms of cumprod and cumsum,
  # approximately as (ignoring boundary issues and initial_value):
  #
  # cumsum(decay_prods * sequence) / decay_prods
  # where decay_prods = reverse_cumprod(decay)
  #
  # One reason this hasn't been done is that multiplying then dividing again by
  # products of decays isn't ideal numerically, in particular if any of the
  # decays are zero it results in NaNs.
  with tf.name_scope(name, values=[sequence, decay, initial_value]):
    if sequence_lengths is not None:
      # Zero out sequence and decay beyond sequence_lengths.
      with tf.control_dependencies(
          [tf.assert_equal(sequence.shape[0], decay.shape[0])]):
        mask = tf.sequence_mask(sequence_lengths, maxlen=sequence.shape[0],
                                dtype=sequence.dtype)
        mask = tf.transpose(mask)

      # Adding trailing dimensions to mask to allow for broadcasting.
      to_seq = mask.shape.dims + [1] * (sequence.shape.ndims - mask.shape.ndims)
      sequence *= tf.reshape(mask, to_seq)
      to_decay = mask.shape.dims + [1] * (decay.shape.ndims - mask.shape.ndims)
      decay *= tf.reshape(mask, to_decay)

    sequences = [sequence, decay]
    if reverse:
      sequences = [_reverse_seq(s, sequence_lengths) for s in sequences]

    summed = tf.scan(lambda a, x: x[0] + x[1] * a,
                     sequences,
                     initializer=tf.convert_to_tensor(initial_value),
                     parallel_iterations=1,
                     back_prop=back_prop)
    if reverse:
      summed = _reverse_seq(summed, sequence_lengths)
    return summed


def _reverse_seq(sequence, sequence_lengths=None):
  """Reverse sequence along dim 0.

  Args:
    sequence: Tensor of shape [T, B, ...].
    sequence_lengths: (optional) tensor of shape [B]. If `None`, only reverse
      along dim 0.

  Returns:
    Tensor of same shape as sequence with dim 0 reversed up to sequence_lengths.
  """
  if sequence_lengths is None:
    return tf.reverse(sequence, [0])

  sequence_lengths = tf.convert_to_tensor(sequence_lengths)
  with tf.control_dependencies(
      [tf.assert_equal(sequence.shape[1], sequence_lengths.shape[0])]):
    return tf.reverse_sequence(
        sequence, sequence_lengths, seq_axis=0, batch_axis=1)
