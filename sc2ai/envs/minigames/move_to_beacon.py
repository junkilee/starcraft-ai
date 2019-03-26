import numpy as np
from gym import spaces
from pysc2.lib import actions, features

from ..minigame import MiniGameEnv

class MoveToBeaconDiscreteEnv(MiniGameEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    
    def _process_reward(self, reward, raw_obs):
        raise NotImplementedError

    def _process_observation(self, raw_obs):
        raise NotImplementedError

