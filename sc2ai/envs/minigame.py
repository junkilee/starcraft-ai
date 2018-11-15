import numpy as np
from gym import spaces
from pysc2.lib import actions, features

from sc2env import SC2GameEnv

class MiniGameEnv(SC2GameEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _reset(self):
        obs = super()._reset()
        return _process_observation(self, obs)

    def _step(self, action):
        raw_obs, reward, done, info = super()._step(self, action)
        reward = self._process_reward(reward, raw_obs)
        obs = self._process_observation(raw_obs)
        return obs, reward, done, info

    def _process_reward(self, reward, raw_obs):
        raise NotImplementedError

    def _process_observation(self, raw_obs):
        raise NotImplementedError

    @property
    def action_space(self):
        pass
    
    @property
    def observation_space(self):

