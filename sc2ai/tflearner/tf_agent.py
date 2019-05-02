from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

from sc2ai.env_interface import AgentAction
from sc2ai.features import MapFeature

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
        self.steps = 0
        self.meta = {} # metadata collected from runs

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
                A list of AgentActions. This will be passed back into self.get_feed_dict during training.
            memory:
                An arbitrary object that will be passed into step at the next timestep.
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
    def get_feed_dict(self, states, masks, actions):
        """
        Get the feed dict with values for all placeholders that are dependenceies for the tensors
            `train_values`, and `train_log_probs`.

        :param masks: A numpy array of shape [T, num_actions].
        :param states: A numpy array of shape [T, *state_shape].
        :param actions: A list of action indices with length T.
        :return: The feed dict required to evaluate `train_values` and `train_log_probs`
        """

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

class InterfaceAgent(ActorCriticAgent, ABC):
    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.num_actions = self.interface.num_actions
        self.num_spatial_actions = self.interface.num_spatial_actions
        self.num_select_actions = self.interface.num_select_unit_actions

        # Used in forward pass
        self.mask_input = tf.placeholder(tf.float32, [None, self.interface.num_actions], name="mask_input")  # [batch, num_actions]
        self.map_features = tf.placeholder(tf.float32, [None, *self.interface.features_shape[MapFeature]], name="map_features") # [batch, *map_features.shape]

        # self.state_input = tf.placeholder(tf.float32, [None, *input_shape])  # [batch, *state_shape]

        # Used in backward pass
        self.action_input = tf.placeholder(tf.int32, [None], name="action_input")  # [T]
        self.spatial_input = tf.placeholder(tf.int32, [None, 2], name="spatial_input")  # [T, 2]   dimension size 2 for x and y
        self.unit_selection_input = tf.placeholder(tf.int32, [None], name="unit_selection_input")

    def get_feed_dict(self, features, masks, actions=None):
        map_features = np.stack([f[MapFeature] for f in features], axis=0)

        feed_dict = {
            self.map_features: map_features,
            self.mask_input: np.array(masks),
        }
        if actions is not None: # backward pass only
            nonspatial, spatials, selection_coords, selection_indices = zip(*[a.as_tuple() for a in actions])
            spatials = [(13, 27) if spatial is None else spatial for spatial in spatials]
            selections = [-1 if selection is None else selection for selection in selection_indices]
            feed_dict[self.action_input] = np.array(nonspatial)
            feed_dict[self.spatial_input] = np.array(spatials)
            feed_dict[self.unit_selection_input] = np.array(selections)
        return feed_dict

    def _get_chosen_selection_probs(self, selection_probs, selection_choice):
        """
        :param selection_probs: Tensor of integers of shape [T, num_units, num_selection_actions]
        :param selection_choice: Tensor of shape [T] of type int
        :return:
        """
        selection_probs = util.index(selection_probs, selection_choice)  # [T, num_selection_actions]
        num_selection_actions = self.interface.num_select_unit_actions()

        index = (self.action_input - self.num_spatial_actions) % tf.convert_to_tensor(num_selection_actions)
        return util.index(selection_probs, index)  # [T]

    def _train_log_probs(self, nonspatial_probs, spatial_probs=None, selection_probs=None):
        nonspatial_log_probs = tf.log(util.index(nonspatial_probs, self.action_input) + 1e-10)

        result = nonspatial_log_probs
        if spatial_probs is not None:
            probs_y = self._get_chosen_spatial_prob(spatial_probs[0], self.spatial_input[:, 1])
            probs_x = self._get_chosen_spatial_prob(spatial_probs[1], self.spatial_input[:, 0])
            spatial_log_probs = tf.log(probs_x + 1e-10) + tf.log(probs_y + 1e-10)
            result = result + tf.where(self.action_input < self.num_spatial_actions,
                                       x=spatial_log_probs,
                                       y=tf.zeros_like(spatial_log_probs))

        if selection_probs is not None:
            probs_selection = self._get_chosen_selection_probs(selection_probs, self.unit_selection_input)
            selection_log_prob = tf.log(probs_selection + 1e-10)
            is_select_action = tf.logical_and(self.action_input >= self.num_spatial_actions,
                                              self.action_input < self.num_spatial_actions + self.num_select_actions)
            result = result + tf.where(is_select_action,
                                       x=selection_log_prob,
                                       y=tf.zeros_like(selection_log_prob))
        return result

    def _get_chosen_spatial_prob(self, spatial_probs, spatial_choice):
        spatial_probs = util.index(spatial_probs, spatial_choice)  # [T, num_screen_dimensions]
        return util.index(spatial_probs, self.action_input % tf.convert_to_tensor(self.num_spatial_actions))  # [T]

    def _probs_from_features(self, features):
        num_steps = tf.shape(self.mask_input)[0]
        nonspatial_probs = parts.actor_nonspatial_head(features[:num_steps], self.mask_input, self.num_actions)
        spatial_probs = parts.actor_spatial_head(features[:num_steps], screen_dim=84, num_spatial_actions=self.num_spatial_actions)
        return nonspatial_probs, spatial_probs

class ConvAgent(InterfaceAgent):
    def __init__(self, interface):
        super().__init__(interface)
        self.features = parts.conv_body(self.map_features, filters=(4,8,), kernel_sizes=(3,3,), strides=(2,2,2,))
        self.nonspatial_probs, self.spatial_probs = self._probs_from_features(self.features)

        self.variable_summaries(self.features)
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('tboard/train', self.session.graph)

    def step(self, features, mask, memory):
        # Collect features across batch
        map_features = np.stack([f[MapFeature] for f in features], axis=0)

        summary, nonspatial_probs, spatial_probs, meta_map_features= self.session.run(
            [self.merged_summary, self.nonspatial_probs, self.spatial_probs, self.map_features], {
                self.map_features: map_features,
                self.mask_input: mask,
            })

        self.train_writer.add_summary(summary, self.steps)

        self.meta['meta_map_features'] = meta_map_features # [batch,x,y,channels]
        # self.meta['meta_final_conv'] = meta_final_conv 
        x_probs = np.expand_dims(spatial_probs[0,0,:,0], axis=0) #[batch, 84] 
        y_probs = np.expand_dims(spatial_probs[1,0,:,0], axis=-1) #[84, batch] 
        self.meta['meta_spatial_probs'] = np.expand_dims(x_probs * y_probs, axis=0)
        
        self.steps += 1

        return util.sample_action(self.num_spatial_actions, nonspatial_probs, spatial_probs), None

    def train_log_probs(self):
        return self._train_log_probs(self.nonspatial_probs, spatial_probs=self.spatial_probs)

    def train_values(self):
        return parts.value_head(self.features)

# class LSTMAgent(InterfaceAgent):
#     def __init__(self, interface):
#         super().__init__(interface)
#         self.num_select_actions = self.interface.num_select_unit_actions

#         self.rnn_size = 256
#         self.self_attention = SelfAttention(hidden_size=128, num_heads=2, attention_dropout=0, train=True)
#         self.lstm = tf.contrib.rnn.LSTMCell(self.rnn_size)

#         self.memory_input = tf.placeholder(tf.float32, [2, None, self.rnn_size], name="memory_input")
#         self.unit_embeddings_input = tf.placeholder(tf.float32, [None, None, self.interface.unit_embedding_size],
#                                                     name="unit_embeddings_input")

#         # TODO: Add in previous action index as an input
#         self.prev_action_input = tf.placeholder(tf.int32, [None], name='prev_action_input')

#         self.features = self.features()  # Shape [batch_size, num_features]

#         lstm_output, self.next_lstm_state = self._lstm_step()
#         self.train_output = self._lstm_step_train()
#         self.all_values = parts.value_head(self.train_output)

#         self.nonspatial_probs, self.spatial_probs_x, self.spatial_probs_y = self._probs_from_features(lstm_output)
#         self.nonspatial_train, self.spatial_train_x, self.spatial_train_y = \
#             self._probs_from_features(self.train_output[:-1])
#         self.unit_selection_probs = self._selection_probs_from_features(lstm_output, self.unit_embeddings_input)
#         self._f1 = lstm_output
#         self.unit_selection_probs_train = self._selection_probs_from_features(self.train_output[:-1],
#                                                                               self.unit_embeddings_input[:-1])

#     def _selection_probs_from_features(self, features, embeddings):
#         return parts.actor_pointer_head(features, embeddings, self.num_select_actions)

#     def features(self):
#         conv_features = parts.conv_body(self.state_input)
#         unit_features = tf.reduce_sum(self.self_attention(self.unit_embeddings_input, bias=0), axis=1)
#         return tf.concat([conv_features, unit_features], axis=1)

#     def step(self, states, masks, memory):
#         """
#         :param states: List of states of length batch size. In this case, state is a dict with keys:
#             "unit_embeddings": numpy array with shape [num_units, embedding_size]
#             "state": numpy array with shape [*state_shape]
#         :param masks: numpy array of shape [batch_size, num_actions]
#         :param memory: numpy of shape [2, batch_size, memory_size] or None for the first step
#         """
#         if memory is None:
#             memory = np.zeros((2, len(states), self.rnn_size))

#         feed_dict = {
#             **self.get_feed_dict(states, masks),
#             self.memory_input: memory
#         }
#         results = self.session.run(
#             [self.next_lstm_state, self.nonspatial_probs, self.unit_selection_probs,
#              *self.spatial_probs_x, *self.spatial_probs_y], feed_dict)
#         next_lstm_state, nonspatial_probs, selection_probs = results[:3]
#         spatial_probs = results[3:]

#         spatial_probs_x = spatial_probs[:self.num_screen_dims]
#         spatial_probs_y = spatial_probs[self.num_screen_dims:]

#         unit_coords = util.pad_stack([state['unit_coords'][:, :2] for state in states], pad_axis=0, stack_axis=0)
#         return self.sample_action_index_with_units(nonspatial_probs, spatial_probs_x,
#                                                    spatial_probs_y, selection_probs, unit_coords), next_lstm_state

#     def _lstm_step(self):
#         state_tuple = tf.unstack(self.memory_input, axis=0)
#         flattened_state = self.features
#         lstm_output, next_state_tuple = self.lstm(flattened_state, state=state_tuple)
#         next_state = tf.stack(next_state_tuple, axis=0)
#         return lstm_output, next_state

#     def _lstm_step_train(self):
#         flattened_state = tf.expand_dims(self.features, axis=0)
#         train_output, _ = tf.nn.dynamic_rnn(self.lstm, flattened_state, dtype=tf.float32)
#         return tf.squeeze(train_output, axis=0)

#     def get_feed_dict(self, states, masks, actions=None):
#         screens = np.stack([state['screen'] for state in states], axis=0)
#         feed_dict = {
#             self.state_input: np.array(states),
#             self.mask_input: np.array(masks),
#         }
#         all_states = states if bootstrap_state is None else [*states, bootstrap_state]
#         unit_embeddings = util.pad_stack([state['unit_embeddings'] for state in all_states], pad_axis=0, stack_axis=0)
#         feed_dict[self.unit_embeddings_input] = unit_embeddings

#         if bootstrap_state is not None:
#             bootstrap_screen = np.expand_dims(bootstrap_state['screen'], axis=0)
#             feed_dict[self.state_input] = np.concatenate([screens, bootstrap_screen], axis=0)
#         else:
#             feed_dict[self.state_input] = screens

#         if actions is not None:
#             nonspatial, spatials, selection_coords, selection_indices = zip(*[a.as_tuple() for a in actions])
#             spatials = [(13, 27) if spatial is None else spatial for spatial in spatials]
#             selections = [-1 if selection is None else selection for selection in selection_indices]
#             feed_dict[self.action_input] = np.array(nonspatial)
#             feed_dict[self.spatial_input] = np.array(spatials)
#             feed_dict[self.unit_selection_input] = np.array(selections)
#         return feed_dict

#     def bootstrap_value(self):
#         return self.all_values[-1]

#     def train_values(self):
#         return self.all_values[:-1]

#     def train_log_probs(self):
#         return self._train_log_probs_with_units(self.nonspatial_train, self.spatial_train_x, self.spatial_train_y,
#                                                 self.unit_selection_probs_train)

#     def sample_action_index_with_units(self, nonspatial_probs, spatial_probs_x, spatial_probs_y,
#                                        unit_distribution, unit_coords):
#         """
#         unit_distribution is an array of shape [num_games, num_select_actions, num_units]
#         :return: Generates an list of action index tuple of type (nonspatial_index, (spatial_x, spatial_y))
#         """
#         actions = self.sample_action_index(nonspatial_probs, spatial_probs_x, spatial_probs_y)
#         num_units = unit_distribution.shape[2]
#         new_actions = []
#         unit_coords = np.stack(unit_coords)

#         for i, action in enumerate(actions):
#             param_type, param_index = self.interface[action.index].param_type
#             if param_type is ActionParamType.SELECT_UNIT:
#                 unit_choice = np.random.choice(num_units, p=unit_distribution[i, param_index])
#                 new_actions.append(AgentAction(
#                                     unit_selection_coords=unit_coords[i, unit_choice].astype(np.int32)))
#             else:
#                 new_actions.append(action)
#         return new_actions

