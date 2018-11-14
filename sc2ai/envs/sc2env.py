import gym
from gym.utils import closer
env_closer = closer.Closer()

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.env.environment import StepType

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

    def _initialize_sc2env(self):
        self._sc2_env = sc2_env.SC2Env(**self._kwargs)

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

    def _seed(self, seed=None):
        pass
    
    def _step(self, action):
        pass

    def _reset(self):
        pass
    
    def _render(self, mode='human', close=False): 
        pass

    def _close(self):
        pass
