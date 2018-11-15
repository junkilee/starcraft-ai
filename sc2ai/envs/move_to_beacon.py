import numpy as np
from gym import spaces
from pysc2.lib import actions, features

from minigame import MiniGameEnv

class MoveToBeaconEnv(MiniGameEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
