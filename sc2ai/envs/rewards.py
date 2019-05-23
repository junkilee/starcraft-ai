from pysc2.lib import actions
from sc2ai.envs import game_info
from gym.spaces.multi_discrete import MultiDiscrete
import logging

logger = logging.getLogger(__name__)


class RewardProcessor:
    def process(self, rew, observation):
        return rew
