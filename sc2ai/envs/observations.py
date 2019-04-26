"""A set of observation filters which convert observation features from pysc2 to processed numpy arrays
"""
from pysc2.lib.features import PlayerRelative, FeatureUnit
from abc import ABC, abstractmethod

class ObservationList():
    def __init__(self, observationlist):
        pass

    def generate_observation(self):
        """

        Returns:

        """
        return

    def convert_to_gym_observation_spaces(filterlist):
        gym_space = None

        for filter in filterlist:
            pass

        return gym_space

class ObservationFilter(ABC):
    """

    """
    def __init__(self, **kwargs):
        return None

    @abstractmethod
    def filter(self, observation):
        """

        Args:
            observation: The observation at each step to be filtered

        Returns:

        """
        pass


class PlayerRelativeLocationMap(ObservationFilter):
    """

    """
    def __init__(self, **kwargs):
        super().__init(**kwargs)

    def filter(self, observation):
        pass


class SelfUnitLocations(ObservationFilter):
    """

    """
    def __init__(self, **kwargs):
        super().__init(**kwargs)

    def filter(self, observation):

        pass
