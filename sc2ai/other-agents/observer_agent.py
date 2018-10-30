# This agent is especially implemented to print out observations while taking random actions.

import numpy

from pysc2.agents import random_agent
from pysc2.agents.scripted_agent import _xy_locs
from pysc2.lib import features
from pysc2.lib import actions

class ObserverAgent(random_agent.RandomAgent):
    """An agent which prints out necessary observations while taking random actions."""

    def step(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        neutral_objects = player_relative == features.PlayerRelative.NEUTRAL
        print(neutral_objects)
        # = _xy_locs(player_relative == _PLAYER_NEUTRAL)

        return super(ObserverAgent, self).step(obs)
