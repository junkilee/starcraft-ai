"""A set of observation filters which convert observation features from pysc2 to processed numpy arrays
"""
from pysc2.lib.features import PlayerRelative, FeatureUnit


def convert_to_gym_observation_spaces(filterlist):
    gym_space = None

    for filter in filterlist:
        pass

    return gym_space

