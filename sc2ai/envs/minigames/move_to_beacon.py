import numpy as np
from gym import spaces
from pysc2.lib import actions, features

from ..sc2env import SingleAgentSC2Env
from ..actions import *
from ..observations import *


class MoveToBeaconEnv(SingleAgentSC2Env):
    """A class containing specifications for the MoveToBeacon Minimap
    """
    def __init__(self, **kwargs):
        action_set = DefaultActionSet([
            NoOpAction(),
            SelectPointAction(select_point_act="select"),
            SelectRectAction(select_add="select"),
            SelectArmyAction(select_add="select"),
            MoveScreenAction(queued="now"),
        ])

        observation_set = CategorizedObservationSet([
            FeatureScreenSelfUnitFilter(),
            FeatureScreenNeuralUnitFilter()
        ])

        super().__init__("MoveToBeacon", action_set, observation_set, **kwargs)
