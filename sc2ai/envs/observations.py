"""A set of observation filters which convert observation features from pysc2
   to processed numpy arrays
"""
from pysc2.lib import features
from pysc2.lib.named_array import NamedNumpyArray
from abc import ABC, abstractmethod
import numpy as np
from sc2ai.envs import game_info
import logging
from gym.spaces.dict import Dict
from gym.spaces.box import Box
from gym.spaces.tuple import Tuple


logger = logging.getLogger(__name__)


class ObservationSet:
    """Default Observation Set containing different categories of filters.
    """
    def __init__(self, categories_list):
        self._categories = categories_list

    def transform_observation(self, observation):
        output_dict = {}
        for category in self._categories:
            output_dict[category.name] = category.transform_observation(observation)
        return output_dict

    def convert_to_gym_observation_spaces(self):
        output_dict = {}
        for category in self._categories:
            output_dict[category.name] = category.convert_to_gym_observation_spaces()
        return Dict(output_dict)


class Category(ABC):
    """An observation set which outputs a Dict gym space.
    """
    def __init__(self, name, filters_list):
        self._name = name
        self._filters = filters_list

    @property
    def name(self):
        return self._name

    @abstractmethod
    def transform_observation(self, observation):
        pass

    @abstractmethod
    def convert_to_gym_observation_spaces(self):
        pass


class MapCategory(Category):
    """A category for handling filters which handle 2D maps.
    It is assumed that filters in the same category share the same 2D dimension.
    """
    def __init__(self, name, filters_list, use_stacked=True):
        super().__init__(name, filters_list)
        self._use_stacked = use_stacked
        # Check if all 2D maps have same dimensions
        if self._use_stacked:
            fixed_space = None
            for _filter in self._filters:
                space = _filter.get_space()
                if fixed_space is None:
                    fixed_space = space
                elif not np.array_equal(fixed_space, space):
                    raise Exception("Filters do not share same dimension.")

    def transform_observation(self, observation):
        if self._use_stacked:
            output = []
            for f in self._filters:
                filtered = f(observation)
                if isinstance(filtered, NamedNumpyArray):
                    filtered = filtered.view(np.ndarray)
                output += (filtered,)
            if len(output) > 1:
                output = np.stack(output)
            else:
                output = output[0]
        else:
            output = {}
            for f in self._filters:
                output[f.name] = f(observation)
        return output

    def convert_to_gym_observation_spaces(self):
        if self._use_stacked:
            shape = self._filters[0].get_space()
            if len(self._filters) > 1:
                shape = np.concatenate([[len(self._filters)], shape])
            return Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
        else:
            output = {}
            for f in self._filters:
                output[f.name] = Box(low=0.0, high=1.0, shape=f.get_space(), dtype=np.float32)
            return Dict(output)

    def __repr__(self):
        return self._name


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

    @abstractmethod
    def __call__(self, observation):
        pass


class FeatureMinimapFilter(ObservationFilter):
    """An abstract class for feature minimap filters.
    Filters a feature minimap information into something meaningful and outputs a same size map with processed data.
    """
    def __init__(self, name, feature_minimap_size = game_info.feature_minimap_size):
        self.feature_minimap_size = feature_minimap_size
        super().__init__("feature_minimap", name)

    def get_space(self):
        return np.array((self.feature_minimap_size, self.feature_minimap_size))

    @abstractmethod
    def __call__(self, observation):
        pass


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
        super().__init__("self_unit", filter_value=features.PlayerRelative.SELF)


class FeatureScreenEnemyUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out enemy units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("enemy_unit", filter_value=features.PlayerRelative.ENEMY)


class FeatureScreenNeutralUnitFilter(FeatureScreenPlayerRelativeFilter):
    """Filters out neutral units as ones and otherwise zeros"""
    def __init__(self):
        super().__init__("neutral_unit", filter_value=features.PlayerRelative.NEUTRAL)


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
