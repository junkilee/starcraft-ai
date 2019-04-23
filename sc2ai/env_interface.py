from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from pysc2.lib import actions as pysc2_actions
from pysc2.lib import static_data
from pysc2.lib.features import PlayerRelative, FeatureUnit

from sc2ai.extract_state import PlayerRelativeMapExtractor, UnitPositionsExtractor, UnitTypeExtractor, HealthMapExtractor

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

    def has_type(self, action_type):
        return self.param_type == action_type

class EnvironmentInterface(ABC):
    def __init__(self):
        self.state_extractors = self._state_extractors()
        self.actions = self._actions()

        self.num_actions = len(self.actions)
        self.num_spatial_actions = len([action for action in self.actions if action.has_type(ActionParamType.SPATIAL)])
        self.num_select_unit_actions = len([action for action in self.actions if action.has_type(ActionParamType.SELECT_UNIT)])
        self.state_shape = self._state_shape()

    def dummy_state(self):
        states = [extractor.dummy_state() for extractor in self.state_extractors]
        return states, np.ones(self.num_actions)

    def _get_action_mask(self, timestep):
        mask = np.ones([self.num_actions])
        for i, action in enumerate(self.actions):
            if action.get_id() not in timestep.observation.available_actions:
                mask[i] = 0
        return mask

    def _state_shape(self):
        shapes = [extractor.shape() for extractor in self.state_extractors]
        return shapes

    def convert_action(self, action):
        return self.actions[action.index].get_sc2_action(action)


    def convert_state(self, timestep):
        """
        :param timestep: Timestep obtained from pysc2 environment step.
        :return: Tuple of converted state (shape self.state_shape) and action mask
        """
        states = [extractor.convert_state(timestep) for extractor in self.state_extractors]
        return states, self._get_action_mask(timestep)

    # ------------------ overrides ------------------
    
    @abstractmethod
    def _state_extractors(self):
        """
        :return: List[StateExtractor]
        """
        pass

    @abstractmethod
    def _actions(self):
        pass

class RoachesEnvironmentInterface(EnvironmentInterface):

    def _actions(self):
        return [
            EnvAction(pysc2_actions.FUNCTIONS.Attack_screen, ActionParamType.SPATIAL, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.Move_screen, ActionParamType.SPATIAL, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.Attack_screen, ActionParamType.SELECT_UNIT, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.select_point, ActionParamType.SELECT_UNIT, ('select', None)),
            EnvAction(pysc2_actions.FUNCTIONS.select_army, ActionParamType.NO_PARAMS, ('select', None)),
        ]

    def _state_extractors(self):
        return [
            PlayerRelativeMapExtractor(),
            HealthMapExtractor(),
        ]

class TrainMarines(RoachesEnvironmentInterface):

    def _actions(self):
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

    def _state_extractors(self):
        return [
            PlayerRelativeMapExtractor(),
        ]

class BeaconEnvironmentInterface(EnvironmentInterface):

    def _actions(self):
        return [
            EnvAction(pysc2_actions.FUNCTIONS.Attack_screen, ActionParamType.SPATIAL, ('now', None)),
            EnvAction(pysc2_actions.FUNCTIONS.select_army, ActionParamType.NO_PARAMS, ('select',)),
            EnvAction(pysc2_actions.FUNCTIONS.no_op, ActionParamType.NO_PARAMS, ('select',)),
        ]

    def _state_extractors(self):
        return [
            PlayerRelativeMapExtractor(),
        ]

