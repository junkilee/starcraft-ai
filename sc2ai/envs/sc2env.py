import logging

import gym
from gym.utils import closer

env_closer = closer.Closer()

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.env.environment import StepType

from game_info import ActionIDs

class SC2Env(gym.Env):
    metadata = {'render.modes': [None, 'human']}
    reward_range = (-np.inf, np.inf)
    spec = None

    action_space = None
    observation_space = None

    _owns_render = True

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs
        self._sc2_env = None
        self._available_actions = None
        self._observation_spec = None

    def _init_sc2_env(self):
        self._sc2_env = sc2_env.SC2Env(**self._kwargs)
        self._observation_spec = self._sc2_env.observation_spec()

    # for backward and forward compatibility
    def seed(self, seed=None):
        return self._seed(seed)
    
    def step(self, action):
        return self._step(self, action)

    def reset(self):
        return self._reset()

    def render(self, mode='human', close=False):
        if not close: # then we have to check rendering mode
            modes = self.metadata.get('render.modes', [])
            if len(modes) == 0:
                raise Exception('{} does not support rendering (requested mode: {})'.format(self, mode))
            elif mode not in modes:
                raise Exception('Unsupported rendering mode: {}. (Supported modes for {}: {})'.format(mode, self, modes))
        return self._render(mode=mode, close=close)

    def close(self):
        if not hasattr(self, '_closed') or self._closed:
            return
        if self._owns_render:
            self.render(close=True)

        self._close()
        env_closer.unregister(self._env_closer_id)
        self._closed = True

    @property
    def spec(self):
        return self._spec

    @property
    def observation_spec(self):
        if self._sc2_env is None:
            self._init_sc2_env()
        return self._observation_spec
    
    @property
    def available_actions(self):
        return self._available_actions

    def _seed(self, seed=None):
        self._seed = seed
    
    def _step(self, action):
        """
        action  expects a tuple which has a specification of an action containing a kind and arguments.
        """
        if action[0] not in self._available_actions:
            logging.warning("The chosen action is not available: %s", action)
            action = [ActionIDs.NO_OP]

        try:
            obs = self._sc2_env.step([actions.FunctionCall(action[0], action[1:])])[0]
        except KeyboardInterrupt:
            logging.info("Keyboard Interruption.")
            return None, 0, True, {}
        except Exception:
            logger.exception("An unexpected exception occured.")
            return None, 0, True, {}
        self.available_actions = obs.observation['available_actions']
        reward = obs.reward
        return obs, reward, obs.step_type == StepType.LAST, {}

    def _reset(self):
        if self._sc2_env is None:
            self._init_sc2_env()
        obs = self._sc2_env.reset()[0]
        self.available_actions = obs.observation['available_actions']
        return obs
    
    def _render(self, mode='human', close=False): 
        pass

    def _close(self):
        if self._sc2_env is not None:
            self._sc2_env.close()
        super()._close()
