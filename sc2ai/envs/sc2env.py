import logging

import gym
from gym.utils import closer

env_closer = closer.Closer()

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.env.environment import StepType

from .game_info import default_env_options
from .game_info import ActionIDs
import sc2ai.envs.minigames as minigames

import numpy as np

from baselines import logger

MAP_ENV_MAPPINGS = {
    "DefeatZerglings": minigames.DefeatZerglingsEnv,
    "CollectMineralAndGas": minigames.DefeatCollectMinardAndGasEnv,
    "MoveToBeacon": minigames.MoveToBeaconEnv
}

def make_sc2env(**kwargs):
    """

    Args:
        **kwargs:

    Returns:

    """
    cls = None
    if kwargs["map"] not in MAP_ENV_MAPPINGS:
        raise Exception("The map is unknown and not registered.")
    else:
        cls = MAP_ENV_MAPPINGS[kwargs["map"]]

    return gym.make(cls(**kwargs))

class SC2Env(gym.Env):
    """A gym wrapper for PySC2's Starcraft II environment.

    Args:
        map_name (str):
        **kwargs:
    """
    metadata = {'render.modes': [None, 'human']}
    reward_range = (-np.inf, np.inf)
    spec = None

    action_space = None
    observation_space = None

    _owns_render = True

    def __init__(self, map_name = None, **kwargs):
        super().__init__()
        self._map_name = map_name
        self._env_options = default_env_options._replace(kwargs)
        self._sc2_env = None
        self._available_actions = None
        self._observation_spec = None
        self._seed = None

    def _init_sc2_env(self):
        """
        Initializes

        Returns:

        """
        players = (sc2_env.Agent(sc2_env.Race[self._env_options.agent1_race], self._env_options.agent1_name),
                   sc2_env.Bot(sc2_env.Race[self._env_options.agent2_race], self._env_options.difficulty))

        self._sc2_env = sc2_env.SC2Env(
            map_name=self._map_name,
            players=players,
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=self._env_options.feature_screen_size,
                feature_minimap=self._env_options.feature_minimap_size,
                rgb_screen=self._env_options.rgb_screen_size,
                rgb_minimap=self._env_options.rgb_minimap_size,
                action_space=self._env_options.action_space,
                use_feature_units=self._env_options.use_feature_units,
                use_raw_units=self._env_options.use_raw_units),
            step_mul=self._env_options.step_mul,
            game_steps_per_episode=self._env_options.game_steps_per_episode,
            disable_fog=self._env_options.disable_fog,
            visualize=self._env_options.render)
        self._observation_spec = self._sc2_env.observation_spec()
        self._action_space = None
        self._observation_space = None
        self._available_actions = None


    def render(self, mode='human', close=False):
        """

        :param mode:
        :param close:
        :return:
        """
        if not close: # then we have to check rendering mode
            modes = self.metadata.get('render.modes', [])
            if len(modes) == 0:
                raise Exception('{} does not support rendering (requested mode: {})'.format(self, mode))
            elif mode not in modes:
                raise Exception('Unsupported rendering mode: {}. (Supported modes for {}: {})'.format(mode, self, modes))
        return

    def close(self):
        """

        :return:
        """
        if not hasattr(self, '_closed') or self._closed:
            return
        if self._owns_render:
            self.render(close=True)
        if self._sc2_env is not None:
            self._sc2_env.close()
        self._closed = True
        super().close()

    @property
    def observation_spec(self):
        if self._sc2_env is None:
            self._init_sc2_env()
        return self._observation_spec
    
    @property
    def available_actions(self):
        return self._available_actions

    def seed(self, seed=None):
        """

        :param seed:
        :return:
        """
        self._seed = seed
    
    def step(self, action):
        """

        :param action: Expects a tuple which has a specification of an action containing a kind and arguments.
        :return:
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
            logging.exception("An unexpected exception occured.")
            return None, 0, True, {}
        self._available_actions = obs.observation['available_actions']
        reward = obs.reward
        return obs, reward, obs.step_type == StepType.LAST, {}

    def reset(self):
        if self._sc2_env is None:
            self._init_sc2_env()
        obs = self._sc2_env.reset()[0]
        self._available_actions = obs.observation['available_actions']
        return obs


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
        return self._process_observation(self, obs)

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