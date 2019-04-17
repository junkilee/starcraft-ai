from enum import Enum
import numpy as np
from pysc2.lib import actions as pysc2_actions
from pysc2.lib import static_data
from pysc2.lib.features import PlayerRelative, FeatureUnit
from abc import ABC, abstractmethod


class ActionParamType(Enum):
    SPATIAL = 1
    SELECT_UNIT = 2
    NO_PARAMS = 3


class AgentAction:
    def __init__(self, index, spatial_coords=None, unit_selection_coords=None, unit_selection_index=None):
        self.index = index
        self.unit_selection_index = unit_selection_index
        self.spatial_coords = spatial_coords
        self.unit_selection_coords = unit_selection_coords

    def as_tuple(self):
        return self.index, self.spatial_coords, self.unit_selection_coords, self.unit_selection_index


class EnvAction:
    def __init__(self, sc2_function, param_type, fixed_params=()):
        self.sc2_function = sc2_function
        self.param_type = param_type
        self.fixed_params = fixed_params

    def get_id(self):
        return self.sc2_function.id

    def get_sc2_action(self, action):
        parameters = []
        if self.param_type == ActionParamType.SPATIAL:
            parameters = [action.spatial_coords]
        elif self.param_type == ActionParamType.SELECT_UNIT:
            parameters = [action.unit_selection_coords]

        all_params = list(self.fixed_params)
        for param in parameters:
            all_params[all_params.index(None)] = param

        return [self.sc2_function(*all_params)]


class EnvironmentInterface(ABC):
    state_shape = None

    def convert_states(self, timesteps):
        state_mask_pairs = [self.convert_state(timestep) for timestep in timesteps]
        state_input = np.stack([pair[0] for pair in state_mask_pairs], axis=0)
        mask_input = np.stack([pair[1] for pair in state_mask_pairs], axis=0)
        return state_input, mask_input

    def convert_actions(self, actions):
        return [self.convert_action(action) for action in actions]

    def convert_action(self, action):
        return self.get_actions()[action.index].get_sc2_action(action)

    @abstractmethod
    def convert_state(self, timestep):
        """
        :param timestep: Timestep obtained from pysc2 environment step.
        :return: Tuple of converted state (shape self.state_shape) and action mask
        """
        pass

    @abstractmethod
    def get_actions(self):
        pass

    def dummy_state(self):
        return np.ones(self.state_shape), np.ones((self.num_actions()))

    def num_spatial_actions(self):
        return len([action for action in self.get_actions() if action.param_type == ActionParamType.SPATIAL])

    def num_select_unit_actions(self):
        return len([action for action in self.get_actions() if action.param_type == ActionParamType.SELECT_UNIT])

    def num_actions(self):
        return len(self.get_actions())

    def _get_action_mask(self, timestep):
        mask = np.ones([self.num_actions()])
        for i, action in enumerate(self.get_actions()):
            if action.get_id() not in timestep.observation.available_actions:
                mask[i] = 0
        return mask


class EmbeddingInterfaceWrapper(EnvironmentInterface):
    """
    Wraps an environment interface to add information about unit embeddings to the state. The state becomes a dictionary
    with keys:
        "unit_embeddings": numpy array of shape `[num_units, embedding_size]`, where `num_units` is arbitrary
        "screen": numpy array of shape `[*self.state_shape]`
    """

    def __init__(self, interface):
        self.interface = interface
        self.unit_embedding_size = len(self._get_embedding_columns()) + len(static_data.UNIT_TYPES)
        self.state_shape = self.interface.state_shape

    def get_actions(self):
        return self.interface.get_actions()

    def convert_state(self, timestep):
        state, action_mask = self.interface.convert_state(timestep)
        coord_columns = np.array([
            FeatureUnit.x,
            FeatureUnit.y,
        ])
        return {
            "screen": state,
            "unit_embeddings": self._get_unit_embeddings(timestep, self._get_embedding_columns()),
            "unit_coords": self._get_unit_embeddings(timestep, coord_columns)
        }, self._augment_mask(timestep, action_mask)

    def dummy_state(self):
        num_units = 0
        state, action_mask = self.interface.dummy_state()
        return {
            "screen": state,
            "unit_embeddings": np.zeros((num_units, self.unit_embedding_size)),
            "unit_coords": np.zeros((num_units, 2 + len(static_data.UNIT_TYPES)))
        }, action_mask

    @staticmethod
    def _get_embedding_columns():
        return np.array([FeatureUnit.alliance,
                         FeatureUnit.health,
                         FeatureUnit.shield,
                         FeatureUnit.energy,
                         FeatureUnit.owner,
                         FeatureUnit.is_selected,
                         FeatureUnit.x,
                         FeatureUnit.y])

    def _get_unit_embeddings(self, timestep, useful_columns):
        unit_info = np.array(timestep.observation.feature_units)
        if unit_info.shape[0] == 0:
            # Set to 1 instead of 0 so no empty embedding situation. Zeros are treated as masked so this is okay.
            return np.zeros((1, self.unit_embedding_size))
        adjusted_info = unit_info[:, np.array(useful_columns)]

        num_unit_types = len(static_data.UNIT_TYPES)
        blizzard_unit_type = unit_info[:, FeatureUnit.unit_type]
        valid_units = np.array([t in static_data.UNIT_TYPES for t in blizzard_unit_type])

        pysc2_unit_type = [static_data.UNIT_TYPES.index(t) for t in blizzard_unit_type[valid_units]]
        one_hot_unit_types = np.eye(num_unit_types)[pysc2_unit_type]

        unit_vector = np.concatenate([adjusted_info[valid_units], one_hot_unit_types], axis=-1)
        return unit_vector

    def _augment_mask(self, timestep, mask):
        """
        Sets unit selection actions to 0 in the mask if there are no units remaining.
        """
        embeddings = self._get_unit_embeddings(timestep, self._get_embedding_columns())
        units_exist = len(embeddings[0]) != 0

        if not units_exist:
            mask[self.interface.num_actions():
                 self.interface.num_actions() + self.interface.num_select_unit_actions()] = 0
        return mask


class RoachesEnvironmentInterface(EnvironmentInterface):
    state_shape = [3, 84, 84]

    def get_actions(self):
        return [
            EnvAction(pysc2_actions.FUNCTIONS.Attack_screen, ActionParamType.SPATIAL, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.Move_screen, ActionParamType.SPATIAL, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.Attack_screen, ActionParamType.SELECT_UNIT, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.select_point, ActionParamType.SELECT_UNIT, ('select', None)),
            EnvAction(pysc2_actions.FUNCTIONS.select_army, ActionParamType.NO_PARAMS, ('select', None)),
        ]

    def convert_state(self, timestep):
        feature_screen = timestep.observation.feature_screen
        beacon = (np.array(feature_screen.player_relative) == PlayerRelative.ENEMY).astype(np.float32)
        player = (np.array(feature_screen.player_relative) == PlayerRelative.SELF).astype(np.float32)
        health = np.array(feature_screen.unit_hit_points).astype(np.float32)
        return np.stack([beacon, player, health], axis=0), self._get_action_mask(timestep)


class TrainMarines(RoachesEnvironmentInterface):
    state_shape = [3, 84, 84]

    def get_actions(self):
        return [
            EnvAction(pysc2_actions.FUNCTIONS.Move_screen, ActionParamType.SPATIAL, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.Build_Barracks_screen, ActionParamType.SPATIAL, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.Build_SupplyDepot_screen, ActionParamType.SPATIAL, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.select_point, ActionParamType.SELECT_UNIT, ('select', None)),
            EnvAction(pysc2_actions.FUNCTIONS.select_idle_worker, ActionParamType.NO_PARAMS, ('select',)),
            EnvAction(pysc2_actions.FUNCTIONS.select_army, ActionParamType.NO_PARAMS, ('now',)),
            EnvAction(pysc2_actions.FUNCTIONS.select_rect, ActionParamType.NO_PARAMS, ('select', [0, 0], [83, 83])),
            EnvAction(pysc2_actions.FUNCTIONS.no_op, ActionParamType.NO_PARAMS, ('select',)),
        ]

    def convert_state(self, timestep):
        feature_screen = timestep.observation.feature_screen
        none = (np.array(feature_screen.player_relative) == PlayerRelative.NONE).astype(np.float32)
        neutral = (np.array(feature_screen.player_relative) == PlayerRelative.NEUTRAL).astype(np.float32)
        player = (np.array(feature_screen.player_relative) == PlayerRelative.SELF).astype(np.float32)
        return np.stack([none, neutral, player], axis=0), self._get_action_mask(timestep)


class BeaconEnvironmentInterface(EnvironmentInterface):
    state_shape = [2, 84, 84]

    def get_actions(self):
        return [
            EnvAction(pysc2_actions.FUNCTIONS.Attack_screen, ActionParamType.SPATIAL, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.select_army, ActionParamType.NO_PARAMS, ('select',)),
        ]

    def convert_state(self, timestep):
        player_relative = timestep.observation.feature_screen.player_relative
        beacon = (np.array(player_relative) == 3).astype(np.float32)
        player = (np.array(player_relative) == 1).astype(np.float32)
        return np.stack([beacon, player], axis=0), self._get_action_mask(timestep)
