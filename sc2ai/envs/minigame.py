import numpy as np
from gym import spaces
from pysc2.lib import actions, features

from sc2env import SC2Env

class MiniGameEnv(SC2Env):
    """Providing a wrapper for minigames. Mainly supports preprocessing of both reward and observation.

    Args:
        map_name:
        **kwargs:
    """
    def __init__(self, map_name, **kwargs):
        assert isinstance(map_name, str)
        super().__init__(map_name, **kwargs)

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

class MultiStepMiniGameEnv(SC2Env):
    """
    Instead of taking one action per step, it handles a list of consecutive actions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _step(self, actions):
        total_reward = 0
        for action in actions:
            raw_obs, reward, done, info = super().__step(self, action)
            total_reward += self._process_reward(reward, raw_obs)
            if done:
                break
        obs = self._process_observation(raw_obs)
        return obs, total_reward, done, info
