import numpy as np
from pysc2.lib import static_data
from pysc2.lib.features import PlayerRelative, FeatureUnit
from abc import ABC, abstractmethod

# ------------------------ Abstract Features ------------------------

class BaseAgentFeature(ABC):
    """ Base Class for any feature used by the agent """
    @abstractmethod
    def extract_from_state(self, timestep):
        pass

    @abstractmethod
    def shape(self): # Shape of Feature
        pass

    def dummy_state(self):
        return np.zeros(self.shape())

class UnitFeature(BaseAgentFeature, ABC):
    """ A (num_units x num_channels) feature tensor """
    pass

class MapFeature(BaseAgentFeature, ABC):
    """ An (84 x 84 x num_channels) feature tensor """
    @abstractmethod
    def _num_channels(self):
        pass

    def shape(self): # override
        return [84,84, self._num_channels()]

    def _channels_last(self, arr): # Input: [channel, x, y]
        return arr.transpose([1, 2, 0]) # Output: [x, y, channel]

# ------------------------ Collection ------------------------

class FeatureCollection:
    """ A collection of features """
    def __init__(self, features):
        self.extractors = {
            MapFeature: CombinedMapFeature([f for f in features if isinstance(f, MapFeature)]),
            UnitFeature: CombinedUnitFeature([f for f in features if isinstance(f, UnitFeature)]),
        }

    def shape(self):
        shapes = {}
        for class_name, extractor in self.extractors.items():
            shapes[class_name] = extractor.shape()

        return shapes

    def extract_from_state(self, timestep):
        """ Each extractor should get its information from state. Return a dict of {Class: Feauture} """
        features = {}
        for class_name, extractor in self.extractors.items():
            features[class_name] = extractor.extract_from_state(timestep)

        return features

class CombinedMapFeature(MapFeature):
    def __init__(self, list_of_features):
        self.features = list_of_features
        self.num_channels = sum([f._num_channels() for f in self.features])

    def _num_channels(self): #override
        return self.num_channels

    def extract_from_state(self, timestep): #override
        return np.concatenate([f.extract_from_state(timestep) for f in self.features], axis=-1)

class CombinedUnitFeature(UnitFeature):
    def __init__(self, list_of_features):
        self.features = list_of_features

    def extract_from_state(self, timestep): #override
        return None

    def shape(self): #override
        return None

# ------------------------ Feature Implementations ------------------------


class PlayerRelativeMapFeature(MapFeature):
    
    def extract_from_state(self, timestep): #override
        player_relative = np.array(timestep.observation.feature_screen.player_relative)
        relative_maps_list = [(player_relative == player).astype(np.float32) for player in PlayerRelative]
        return self._channels_last(np.stack(relative_maps_list, axis=0))

    def _num_channels(self): #override
        return len(PlayerRelative)

class HealthMapFeature(MapFeature):
    
    def extract_from_state(self, timestep): #override
        health = np.array(timestep.observation.feature_screen.unit_hit_points).astype(np.float32)
        return self._channels_last(np.stack([health], axis=0))

    def _num_channels(self): #override
        return 1


class StateExtractor(ABC):

    @abstractmethod
    def convert_state(self, timestep):
        pass

    @abstractmethod
    def shape(self):
        """
        :return List[Int] size of input
        """
        pass

    def dummy_state(self):
        return np.zeros(self.output_shape)

class FeatureUnitExtractor(StateExtractor, ABC):

    """ @override """
    def convert_state(self, timestep):
        columns = self._feature_unit_cols()
        unit_info = np.array(timestep.observation.feature_units)[:,columns]

        return FeatureUnitState(unit_info)

    """ @override """
    def shape(self):
        # [num_units, num_cols]
        num_cols = len(self._feature_unit_cols())
        return (None, num_cols)

    @abstractmethod
    def _feature_unit_cols(self):
        pass

       
class UnitPositionsExtractor(FeatureUnitExtractor):

    """ @override """
    def _feature_unit_cols(self):
        return np.array([
            FeatureUnit.x,
            FeatureUnit.y,
        ])

class UnitTypeExtractor(FeatureUnitExtractor):

    """ @override """
    def convert_state(self, timestep):
        return self._get_one_hot_unit_type(timestep)

    """ @override """
    def shape(self):
        # [num_units, num_cols]
        num_cols = len(static_data.UNIT_TYPES) + 1 # add option for unknown unit
        return (None, num_cols)

    def _get_one_hot_unit_type(self, timestep):
        num_unit_types = self.output_shape()[1]
        blizzard_unit_type = np.array(timestep.observation.feature_units)[:, FeatureUnit.unit_type]
        pysc2_unit_type = np.array(
            [static_data.UNIT_TYPES.index(t) if t in static_data.UNIT_TYPES else 0 for t in blizzard_unit_type]
        )
        one_hot_unit_types = np.eye(num_unit_types)[pysc2_unit_type]

        return one_hot_unit_types

# class PlayerRelativeMapExtractor(StateExtractor):
    
#     """ @override """
#     def convert_state(self, timestep):
#         player_relative = np.array(timestep.observation.feature_screen.player_relative)
#         relative_maps = [(player_relative == player).astype(np.float32) for player in PlayerRelative]
#         return channels_first(np.stack(relative_maps, axis=0))

#     """ @override """
#     def shape(self):
#         # [84, 84, players]
#         return [84,84] + [len(PlayerRelative)]

# class HealthMapExtractor(StateExtractor):
    
#     """ @override """
#     def convert_state(self, timestep):
#         health = np.array(timestep.observation.feature_screen.unit_hit_points).astype(np.float32)
#         return channels_first(np.stack([health], axis=0))

#     """ @override """
#     def shape(self):
#         # [84, 84, 1]
#         return [84, 84, 1]


# def channels_first(arr): # [channel, x, y]
#     return arr.transpose([1, 2, 0])
