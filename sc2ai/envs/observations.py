"""A set of observation filters which convert observation features from pysc2 to processed numpy arrays
"""
from pysc2.lib import features
from abc import ABC, abstractmethod
import numpy as np
import sc2ai as game_info
from gym.spaces.dict_space import Dict
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

class ObservationSet(ABC):
    @abstractmethod
    def generate_observation(self, observation):
        """Generate a gym consumable observation instance from the pysc2 observation output.

        Args:
            observation: a named dict containing named numpy array coming from pysc2.

        Returns:
            a dictionary containing observation in numpy array format.
        """
        pass

    @abstractmethod
    def convert_to_gym_observation_spaces(self):
        """Generate a set of gym observation spaces from the given set of Observation Filters.

        Returns:
            a gym Space representing the given set of the Observation Filters.
        """
        pass


class CategorizedObservationSet(ObservationSet):
    """An observation set which outputs a Dict gym space which

    """
    def __init__(self, filters_list):
        self._filters_list = filters_list
        self._categories = []
        for f in self.filters_list:
            if f.category not in self._categories:
                self._categories += f.category

    def generate_observation(self, observation):
        observation_outputs = []
        output_dict = {}

        for f in self.filters_list:
            observation_outputs.add(f.filter(observation))

        return observation_outputs

    def convert_to_gym_observation_spaces(self):
        observation_dict = {}
        for category in self._categories:
            observation_dict[category] = []
        for f in self.filters_list:
            observation_dict[filter.cateogry] += [f.get_space()]
        return Dict()


class ObservationFilter(ABC):
    """An abstract class for every observation filer.

    Currently the filter set only supports single-depth categories.
    """
    def __init__(self, category, name):
        self._category = category
        self._name = name

    @property
    def category(self):
        return self._category

    @property
    def name(self):
        return self._name

    @abstractmethod
    def __call__(self, observation):
        pass

    @abstractmethod
    def get_space(self):
        pass


class FeatureScreenFilter(ObservationFilter):
    """An abstract class for feature screen filters.
    Filters a feature screen information into something meaningful and outputs a same size map with processed data.
    """
    def __init__(self, name, feature_screen_size = game_info.feature_screen_size):
        self._feature_screen_size = feature_screen_size
        super().__init__("feature_screen", name)

    def get_space(self):
        return np.array([self._feature_screen_size, self._feature_screen_size])

    def __call__(self, observation):
        raise NotImplementedError()


class FeatureMinimapFilter(ObservationFilter):
    """An abstract class for feature minimap filters.
    Filters a feature minimap information into something meaningful and outputs a same size map with processed data.
    """
    def __init__(self, name, feature_minimap_size = game_info.feature_minimap_size):
        self.feature_minimap_size = feature_minimap_size
        super().__init__("feature_minimap", name)

    def get_space(self):
        return np.array((self.feature_minimap_size, self.feature_minimap_size]))

    def __call__(self, observation):
        raise NotImplementedError()


class FeatureScreenPlayerRelativeFilter(FeatureScreenFilter):
    """Outputs a filtered map of the player relative information"""
    def __init__(self, name, filter_value):
        super().__init__(name)
        self._filter_value = filter_value

    def _filter(self, observation):
        player_relative = observation.feature_screen.player_relative
        return (player_relative == self._filter_value).astype(np.float32)

    def __call__(self, observation):
        return self._filter(observation)

class FeatureScreenSelfUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out self units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("Feature-Screen-Self-Unit", filter_vallue=features.PlayerRelative.SELF)


class FeatureScreenEnemyUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out enemy units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("Feature-Screen-Self-Unit", filter_vallue=features.PlayerRelative.ENEMY)


class FeatureScreenNeuralUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out neutral units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("Feature-Screen-Self-Unit", filter_vallue=features.PlayerRelative.NEUTRAL)


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