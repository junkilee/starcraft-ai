from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from pysc2.lib import actions as pysc2_actions
from pysc2.lib import static_data
from pysc2.lib.features import PlayerRelative, FeatureUnit

from sc2ai.features import *

# -------------------------------- AgentAction --------------------------------

class AgentAction:
    def __init__(self, index, spatial_coords=None, unit_selection_coords=None, unit_selection_index=None):
        self.index = index
        self.unit_selection_index = unit_selection_index
        self.spatial_coords = spatial_coords
        self.unit_selection_coords = unit_selection_coords

    def as_tuple(self):
        return self.index, self.spatial_coords, self.unit_selection_coords, self.unit_selection_index

# -------------------------------- EnvAction --------------------------------

class BaseEnvAction(ABC):
    def __init__(self, sc2_function, fixed_params=()):
        self.sc2_function = sc2_function
        self.fixed_params = fixed_params

    def get_id(self):
        return self.sc2_function.id

    def get_sc2_action(self, agent_action):
        """ Converts an AgentAction to an Sc2Action """
        all_params = list(self.fixed_params)
        for param in self.dynamic_params(agent_action):
            all_params[all_params.index(None)] = param

        return [self.sc2_function(*all_params)]

    @abstractmethod
    def dynamic_params(self, agent_action):
        """ Extracts the relavent parts from the AgentAction for this action """
        pass

class NoParamEnvAction(BaseEnvAction):
    def dynamic_params(self, agent_action): #override
        return []

class SpatialEnvAction(BaseEnvAction):
    def dynamic_params(self, agent_action): #override
        return [agent_action.spatial_coords]

class SelectUnitEnvAction(BaseEnvAction):
    def dynamic_params(self, agent_action): #override
        return [agent_action.unit_selection_coords]

# -------------------------------- EnvironmentInterface --------------------------------

class EnvironmentInterface(ABC):
    def __init__(self):
        self.feature_collection = FeatureCollection(self._features())
        self.actions = self._actions()

        self.num_actions = len(self.actions)
        self.num_spatial_actions = len([action for action in self.actions if isinstance(action, SpatialEnvAction)])
        self.num_select_unit_actions = len([action for action in self.actions if isinstance(action, SelectUnitEnvAction)])
        self.state_shape = self._state_shape()

    # ---------- Action ----------

    def _get_action_mask(self, timestep):
        mask = np.ones([self.num_actions])
        for i, action in enumerate(self.actions):
            if action.get_id() not in timestep.observation.available_actions:
                mask[i] = 0
        return mask

    def to_env_action(self, agent_action):
        return self.actions[agent_action.index].get_sc2_action(agent_action)

    # ---------- State ----------

    def dummy_state(self):
        state = self.feature_collection.dummy_state()
        return state, np.ones(self.num_actions)

    def _state_shape(self):
        return self.feature_collection.shape()

    def to_features(self, timestep):
        """
        :param timestep: Timestep obtained from pysc2 environment step.
        :return: Tuple of converted state (shape self.state_shape) and action mask
        """
        extracted_features = self.feature_collection.extract_from_state(timestep)
        return extracted_features, self._get_action_mask(timestep)

    # ------------------ overrides ------------------
    
    @abstractmethod
    def _features(self):
        """ List[BaseAgentFeature] """
        pass

    @abstractmethod
    def _actions(self):
        """ List[BaseEnvAction] """
        pass

class RoachesEnvironmentInterface(EnvironmentInterface):

    def _actions(self):
        return [
            SpatialEnvAction(pysc2_actions.FUNCTIONS.Attack_screen, ('now', None)),
            SpatialEnvAction(pysc2_actions.FUNCTIONS.Move_screen, ('now', None)),

            SelectUnitEnvAction(pysc2_actions.FUNCTIONS.Attack_screen, ('now', None)),
            SelectUnitEnvAction(pysc2_actions.FUNCTIONS.select_point, ('select', None)),

            NoParamEnvAction(pysc2_actions.FUNCTIONS.select_army, ('select', None)),
        ]

    def _features(self):
        return [
            PlayerRelativeMapFeature(),
            HealthMapExtractor(),
        ]

class TrainMarines(RoachesEnvironmentInterface):

    def _actions(self):
        return [
            SpatialEnvAction(pysc2_actions.FUNCTIONS.Move_screen, ('now', None)),
            SpatialEnvAction(pysc2_actions.FUNCTIONS.Build_Barracks_screen, ('now', None)),
            SpatialEnvAction(pysc2_actions.FUNCTIONS.Build_SupplyDepot_screen, ('now', None)),

            SelectUnitEnvAction(pysc2_actions.FUNCTIONS.select_point, ('select', None)),

            NoParamEnvAction(pysc2_actions.FUNCTIONS.select_idle_worker, ('select',)),
            NoParamEnvAction(pysc2_actions.FUNCTIONS.select_army, ('now',)),
            NoParamEnvAction(pysc2_actions.FUNCTIONS.select_rect, ('select', [0, 0], [83, 83])),
            NoParamEnvAction(pysc2_actions.FUNCTIONS.no_op, ('select',)),
        ]

    def _features(self):
        return [
            PlayerRelativeMapFeature(),
        ]

class BeaconEnvironmentInterface(EnvironmentInterface):

    def _actions(self):
        return [
            SpatialEnvAction(pysc2_actions.FUNCTIONS.Attack_screen, ('now', None)),
            NoParamEnvAction(pysc2_actions.FUNCTIONS.select_army, ('select',)),
            NoParamEnvAction(pysc2_actions.FUNCTIONS.no_op, ('select',)),
        ]

    def _features(self):
        return [
            PlayerRelativeMapFeature(),
        ]

