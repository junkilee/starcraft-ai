# Agents are implemented here with the help of below references.
# https://github.com/deepmind/pysc2/tree/master/pysc2/agents
# https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

import numpy

from pysc2.agents import base_agent
from pysc2.agents.base_agent import _xy_locs
from pysc2.lib import features
from pysc2.lib import actions

class NoOpAgent(base_agent.BaseAgent):
    """An agent which does nothing. Please use this as a template in making other agents.
       Use the below command run this agent

       python3 -m pysc2.bin.agent --map MoveToBeacon --agent sc2ai.basic_agents.NoOpAgent
    """
    def step(self, obs):
        super(NoOpAgent, self).step(obs)

        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
