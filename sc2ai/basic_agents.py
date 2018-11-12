# Agents are implemented here with the help of below references.
# https://github.com/deepmind/pysc2/tree/master/pysc2/agents
# https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

import numpy

from pysc2.agents import base_agent
from pysc2.agents.scripted_agent import _xy_locs
from pysc2.lib import features
from pysc2.lib import actions

class NoOpAgent(base_agent.BaseAgent):
    """An agent which does nothing. Please use this as a template in making other agents.
       Use the below command run this agent

       python3 -m pysc2.bin.agent --map MoveToBeacon --agent sc2ai.basic_agents.NoOpAgent
    """
    def step(self, obs):
        super(NoOpAgent, self).step(obs)
        print("Raw Units---")
        print(obs.observation.raw_units)

        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

def get_self_entities_locations(obs):
    self_entities = obs.observation.feature_screen.player_relative == features.PlayerRelative.SELF
    return _xy_locs(self_entities)

def get_neutral_entities_locations(obs):
    self_entities = obs.observation.feature_screen.player_relative == features.PlayerRelative.NEUTRAL
    return _xy_locs(self_entities)

def get_enemy_entities_locations(obs):
    self_entities = obs.observation.feature_screen.player_relative == features.PlayerRelative.ENEMY
    return _xy_locs(self_entities)

class MoveToBeacon(base_agent.BaseAgent):
    def step(self, obs):
        #print('wewerwerwer')
        #print(obs.observation.raw_units)
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            marines = get_self_entities_locations(obs)
            marine_xy = numpy.mean(marines, axis=0).round()
            beacons = get_neutral_entities_locations(obs)
            if not beacons:
                return actions.FUNCTIONS.no_op()
            closest_distance = numpy.Inf
            goto_xy = [0, 0]
            for beacon in beacons:
                print(beacon)
                print(marine_xy)
                distance = numpy.linalg.norm(numpy.array(beacon) - marine_xy, axis=1)
                if closest_distance > distance:
                    distance = closest_distance
                    goto_xy = beacon

            return actions.FUNCTIONS.Move_screen("now", goto_xy)
        else:
            return actions.FUNCTIONS.select_army("select")

class CollectMineralShards(base_agent.BaseAgent):
    def step(self, obs):
        pass

class DefeatRoaches(base_agent.BaseAgent):
    def step(self, obs):
        pass
