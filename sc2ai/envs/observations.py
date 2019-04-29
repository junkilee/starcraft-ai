"""A set of observation filters which convert observation features from pysc2 to processed numpy arrays
"""
from pysc2.lib import features
from abc import ABC, abstractmethod
import numpy as np
import sc2ai as game_info
from gym.spaces.multi_discrete import MultiDiscrete

class ObservationList:
    def __init__(self, filters_list):
        self.filters_list = filters_list

    def generate_observation(self, observation):
        observation_outputs = []

        for filter in self.filters_list:
            observation_outputs.add(filter.filter(observation))

        return observation_outputs

    def convert_to_gym_observation_spaces(self):
        filters_list = []

        for filter in self.filters_list:
            filters_list.add(filter.get_space())

        return MultiDiscrete()

class ObservationFilter(ABC):
    """An abstract class for every observation filer."""
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_space(self):
        pass

    @abstractmethod
    def filter(self, observation):
        pass


class FeatureScreenFilter(ObservationFilter):
    """An abstract class for feature screen filters.
    Filters a feature screen information into something meaningful and outputs a same size map with processed data.
    """
    def __init__(self, name, feature_screen_size = game_info.feature_screen_size):
        self.feature_screen_size = feature_screen_size
        super().__init__(name)

    def get_space(self):
        return [self.feature_screen_size, self.feature_screen_size]

    def filter(self, observation):
        raise NotImplementedError()

class FeatureMinimapFilter(ObservationFilter):
    """An abstract class for feature minimap filters.
    Filters a feature minimap information into something meaningful and outputs a same size map with processed data.
    """
    def __init__(self, name, feature_minimap_size = game_info.feature_minimap_size):
        self.feature_minimap_size = feature_minimap_size
        super().__init__(name)

    def get_space(self):
        return [self.feature_minimap_size, self.feature_minimap_size]

    def filter(self, observation):
        raise NotImplementedError()

class FeatureScreenPlayerRelativeFilter(FeatureScreenFilter):
    """Outputs a filtered map of the player relative information"""
    def __init__(self, name):
        super().__init__(name)

    def filter(self, observation, filter_value):
        """Filters out a filter value from the features player relative array.

        Args:
            observation: a named list of numpy arrays coming from the environment's step method.
            filter_value: a value for filtering the feature plane's player relative map. If it matches the value then
                          output is 1 otherwise 0.

        Returns:
            A numpy array
        """
        player_relative = observation.feature_screen.player_relative
        output = (player_relative == filter_value).astype(np.float32)
        return output


class FeatureScreenSelfUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out self units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("Feature-Screen-Self-Unit")

    def filter(self, observation):
        return super().filter(observation, filter_vallue=features.PlayerRelative.SELF)


class FeatureScreenEnemyUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out enemy units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("Feature-Screen-Self-Unit")

    def filter(self, observation):
        return super().filter(observation, filter_vallue=features.PlayerRelative.ENEMY)

class FeatureScreenNeuralUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out neutral units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("Feature-Screen-Self-Unit")

    def filter(self, observation):
        return super().filter(observation, filter_vallue=features.PlayerRelative.NEUTRAL)

class FeatureScreenUnitHitPointFilter(FeatureScreenFilter):
    """Filters out neutral units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("Feature-Screen-Self-Unit")

    def filter(self, observation):
        """Filters out a filter value from the features player unit hit point (HP) array.
        We especially pick a ratio version of this and then rescale for the values to stay between zero and one.

        Args:
            observation: a named list of numpy arrays coming from the environment's step method.

        Returns:
            A numpy array
        """
        return observation.feature_screen.unit_hit_points_ratio / 256.0