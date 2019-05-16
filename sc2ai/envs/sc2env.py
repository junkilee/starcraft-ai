import numpy as np
import logging
import gym
from gym.utils import closer

logger = logging.getLogger(__name__)
env_closer = closer.Closer()

from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from .game_info import default_env_options
import sc2ai.envs.minigames as minigames
from sc2ai.envs.rewards import RewardProcessor


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
    if kwargs["map"] not in MAP_ENV_MAPPINGS:
        raise Exception("The map is unknown and not registered.")
    else:
        cls = MAP_ENV_MAPPINGS[kwargs["map"]]

    return gym.make(cls(**kwargs))

class SingleAgentSC2Env(gym.Env):
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

    def __init__(self, map_name, action_set, observation_set, reward_processor=RewardProcessor(), **kwargs):
        super().__init__()
        self._map_name = map_name
        self._env_options = default_env_options._replace(kwargs)
        self._sc2_env = None
        self._seed = None
        self._observation_spec = None
        self._action_set = action_set
        self._action_space = None
        self._observation_spec = None
        self._observation_set = observation_set
        self._observation_space = None
        self._reward_processor = reward_processor

    def _init_sc2_env(self):
        """
        Initializes the PySC2 environment

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
        self._action_space = self._action_set.convert_to_gym_action_spaces()
        self._observation_space = self._observation_set.convert_to_gym_observation_spaces()

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
    def action_space(self):
        if self._sc2_env is None:
            self._init_sc2_env()
        return self._action_space

    @property
    def observation_space(self):
        if self._sc2_env is None:
            self._init_sc2_env()
        return self._observation_space

    def _process_reward(self, reward, raw_obs):
        return self._reward_processor.process(reward, raw_obs)

    def seed(self, seed=None):
        """

        :param seed:
        :return:
        """
        self._seed = seed

    def step(self, actions):
        total_reward = 0
        transformed_actions = self._action_space.transform_action(actions)
        for action in transformed_actions:
            raw_obs, reward, done, info = self._single_step(self, action)
            total_reward += self._process_reward(reward, raw_obs)
            if done:
                break
        self._action_set.update_available_actions(raw_obs.available_actions)
        obs = self._observation_set.transform_observation(raw_obs)
        return obs, total_reward, done, info
    
    def _single_step(self, action):
        try:
            # Only observing the first player's timestep
            timestep = self._sc2_env.step([action])[0]
        except KeyboardInterrupt:
            logging.info("Keyboard Interruption.")
            return None, 0, True, {}
        except Exception:
            logging.exception("An unexpected exception occured.")
            return None, 0, True, {}
        reward = timestep.reward
        return timestep.observation, reward, timestep.step_type == StepType.LAST, {}

    def reset(self):
        if self._sc2_env is None:
            self._init_sc2_env()
        raw_obs = self._sc2_env.reset()[0].observation
        self._action_set.update_available_actions(raw_obs.available_actions)
        obs = self._observation_set.transform_observation(raw_obs)
        return obs
