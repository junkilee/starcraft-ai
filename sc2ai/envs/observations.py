"""A set of observation filters which convert observation features from pysc2
   to processed numpy arrays
"""
from pysc2.lib import features
from abc import ABC, abstractmethod
import numpy as np
import sc2ai as game_info
import logging
from gym.spaces.dict_space import Dict
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

logger = logging.getLogger(__name__)


class ObservationSet(ABC):
    def __init__(self, filters_list):
        self._filters_list = filters_list

    @abstractmethod
    def transform_observation(self, observation):
        """transform into a gym consumable observation instance from the pysc2 observation output.

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
    """An observation set which outputs a Dict gym space.
    """
    def __init__(self, filters_list, use_stacked = True):
        super().__init__(filters_list)
        self._categories = []
        self._use_stacked = use_stacked
        for f in self._filters_list:
            if f.category not in self._categories:
                self._categories += (f.category,)

    def transform_observation(self, observation):
        output_dict = {}
        if self._use_stacked:
            for category in self._categories:
                output_dict[category] = []
            for f in self._filters_list:
                output_dict[f.category] += (f(observation),)
            for category in self._categories:
                if len(output_dict[category]) > 1:
                    output_dict[category] = np.stack(output_dict[category])
                else:
                    output_dict[category] = output_dict[category][0]
        else:
            for category in self._categories:
                output_dict[category] = {}
            for f in self._filters_list:
                output_dict[f.category][f.name] = f(observation)
        return output_dict

    def convert_to_gym_observation_spaces(self):
        output_dict = {}
        if self._use_stacked:
            for category in self._categories:
                output_dict[category] = []
            for f in self._filters_list:
                output_dict[f.cateogry] += (f.get_space(),)
            for category in self._categories:
                if len(output_dict[category]) > 1:
                    output_dict[category] = np.stack(output_dict[category])
                else:
                    output_dict[category] = output_dict[category][0]
        else:
            for category in self._categories:
                output_dict[category] = {}
            for f in self._filters_list:
                output_dict[f.category][f.name] = f.get_space()
        return output_dict


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
        return np.array((self._feature_screen_size, self._feature_screen_size))

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
        return np.array((self.feature_minimap_size, self.feature_minimap_size))

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
        super().__init__("self_unit", filter_vallue=features.PlayerRelative.SELF)


class FeatureScreenEnemyUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out enemy units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("enemy_unit", filter_vallue=features.PlayerRelative.ENEMY)


class FeatureScreenNeuralUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out neutral units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("neutral_unit", filter_vallue=features.PlayerRelative.NEUTRAL)


class FeatureScreenUnitHitPointFilter(FeatureScreenFilter):
    """Filters out neutral units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("hit_points")

    def _filter(self, observation):
        """Filters out a filter value from the features player unit hit point (HP) array.
        We especially pick a ratio version of this and then rescale for the values to stay between zero and one.

        Args:
            observation: a named list of numpy arrays coming from the environment's step method.

        Returns:
            A numpy array
        """
        return observation.feature_screen.unit_hit_points_ratio / 256.0

    def __call__(self, observation):
        return self._filter(observation)
