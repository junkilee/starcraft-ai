import numpy as np
from pysc2.lib import static_data
from pysc2.lib.features import PlayerRelative, FeatureUnit
from abc import ABC, abstractmethod

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
    """
    Wraps an environment interface to add information about unit embeddings to the state. The state becomes a dictionary
    with keys:
        "unit_embeddings": numpy array of shape `[num_units, embedding_size]`, where `num_units` is arbitrary
        "screen": numpy array of shape `[*self.state_shape]`
    """

    """ @override """
    def convert_state(self, timestep):
        return self._get_unit_info(timestep, self._feature_unit_cols())

    """ @override """
    def shape(self):
        # [num_units, num_cols]
        num_cols = len(self._feature_unit_cols())
        return (None, num_cols)

    @abstractmethod
    def _feature_unit_cols(self):
        pass

    def _get_unit_info(self, timestep, columns):
        return np.array(timestep.observation.feature_units)[:,columns]
       
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

class PlayerRelativeMapExtractor(StateExtractor):
    
    """ @override """
    def convert_state(self, timestep):
        player_relative = np.array(timestep.observation.feature_screen.player_relative)
        relative_maps = [(player_relative == player).astype(np.float32) for player in PlayerRelative]
        return channels_first(np.stack(relative_maps, axis=0))

    """ @override """
    def shape(self):
        # [84, 84, players]
        return [84,84] + [len(PlayerRelative)]

class HealthMapExtractor(StateExtractor):
    
    """ @override """
    def convert_state(self, timestep):
        health = np.array(timestep.observation.feature_screen.unit_hit_points).astype(np.float32)
        return channels_first(np.stack([health], axis=0))

    """ @override """
    def shape(self):
        # [84, 84, 1]
        return [84, 84, 1]


def channels_first(arr): # [channel, x, y]
    return arr.transpose([1, 2, 0])
