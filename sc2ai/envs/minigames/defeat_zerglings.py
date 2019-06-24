from sc2ai.envs.sc2env import SingleAgentSC2Env
from sc2ai.envs.actions import *
from sc2ai.envs.observations import *


class DefeatZerglingsEnv(SingleAgentSC2Env):
    """A class containing specifications for the CollectMineralsAndGas Minimap
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

        super().__init__("DefeatZerglings", action_set, observation_set, **kwargs)