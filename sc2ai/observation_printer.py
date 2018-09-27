# This agent is especially implemented to print out observations while taking random actions.

import numpy

from pysc2.agents import base_agent, random_agent
from pysc2.lib import actions

class ObserverAgent(random_agent.RandomAgent):
    """An agent which prints out necessary observations while taking random actions."""

    def step(self, obs):

        return super(ObserverAgent, self).step(obs)
