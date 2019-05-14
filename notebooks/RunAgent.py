
# coding: utf-8

# In[ ]:


import importlib
import threading
from absl import app
from absl import flags

from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch
from collections import namedtuple
from pysc2.agents import base_agent
from pysc2.agents.scripted_agent import _xy_locs
from pysc2.lib import features
from pysc2.lib import actions
import numpy

SC2EnvOptions = namedtuple('SC2EnvOptions', ('map', 'render', 'feature_screen_size', 'feature_minimap_size',
                                             'rgb_screen_size', 'rgb_minimap_size', 'action_space',
                                             'use_feature_units', 'use_raw_units', 'disable_fog',
                                             'max_agent_steps', 'game_steps_per_episode', 'max_episodes',
                                             'step_mul',
                                             'agent2',
                                             'agent1_name', 'agent1_race', 'agent2_name', 'agent2_race',
                                             'difficulty', 'profile', 'trace', 'parallel', 'save_replay'))
""" The definition of SC2EnvOptions """
options = SC2EnvOptions(map="MoveToBeacon",
                        render=True,
                        feature_screen_size=84,
                        feature_minimap_size=64,
                        rgb_screen_size=None,
                        rgb_minimap_size=None,
                        action_space="features",
                        use_feature_units=True,
                        use_raw_units=True,
                        disable_fog=True,
                        max_agent_steps=0,
                        game_steps_per_episode=None,
                        max_episodes=0,
                        step_mul=80,
                        agent2="Bot",
                        agent1_name="TrainedAI",
                        agent1_race="terran",
                        agent2_name="DefaultAI",
                        agent2_race="terran",
                        difficulty=sc2_env.Difficulty.very_easy,
                        profile=False,
                        trace=False,
                        parallel=1,
                        save_replay=True)

def run_thread(agent_classes, players, map_name, visualize):
  """Run one thread worth of the environment with agents."""
  with sc2_env.SC2Env(
      map_name=map_name,
      players=players,
      agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=options.feature_screen_size,
          feature_minimap=options.feature_minimap_size,
          rgb_screen=options.rgb_screen_size,
          rgb_minimap=options.rgb_minimap_size,
          action_space=options.action_space,
          use_feature_units=options.use_feature_units),
      step_mul=options.step_mul,
      game_steps_per_episode=options.game_steps_per_episode,
      disable_fog=options.disable_fog,
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    agents = [agent_cls() for agent_cls in agent_classes]
    run_loop.run_loop(agents, env, options.max_agent_steps, options.max_episodes)
    if options.save_replay:
      env.save_replay(agent_classes[0].__name__)

agent_class = None

def main(unused_argv):
  """Run an agent."""
  stopwatch.sw.enabled = options.profile or options.trace
  stopwatch.sw.trace = options.trace

  map_inst = maps.get(options.map)

  agent_classes = []
  players = []

  agent_cls = agent_class
  agent_classes.append(agent_cls)
  players.append(sc2_env.Agent(sc2_env.Race[options.agent1_race],
                               options.agent1_name))

  if map_inst.players >= 2:
    if options.agent2 == "Bot":
      players.append(sc2_env.Bot(sc2_env.Race[options.agent2_race],
                                 sc2_env.Difficulty[options.difficulty]))
    else:
      agent_module, agent_name = options.agent2.rsplit(".", 1)
      agent_cls = getattr(importlib.import_module(agent_module), agent_name)
      agent_classes.append(agent_cls)
      players.append(sc2_env.Agent(sc2_env.Race[options.agent2_race],
                                   options.agent2_name or agent_name))

  threads = []
  for _ in range(options.parallel - 1):
    t = threading.Thread(target=run_thread,
                         args=(agent_classes, players, options.map, False))
    threads.append(t)
    t.start()

  run_thread(agent_classes, players, options.map, options.render)

  for t in threads:
    t.join()

  if options.profile:
    print(stopwatch.sw)


class NoOpAgent(base_agent.BaseAgent):
    """An agent which does nothing. Please use this as a template in making other agents.
       Use the below command run this agent

       python3 -m pysc2.bin.agent --map MoveToBeacon --agent sc2ai.basic_agents.NoOpAgent
    """
    def step(self, obs):
        super(NoOpAgent, self).step(obs)

        ##--------------------------------------
        print(obs.observation.available_actions)
        ##--------------------------------------

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
        print(obs.observation.available_actions)
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            marines = get_self_entities_locations(obs)
            marine_xy = numpy.mean(marines, axis=0).round()
            beacons = get_neutral_entities_locations(obs)
            if not beacons:
                return actions.FUNCTIONS.no_op()
            closest_distance = numpy.Inf
            goto_xy = [0, 0]
            for beacon in beacons:
                distance = numpy.linalg.norm(numpy.array(beacon) - marine_xy)
                if closest_distance > distance:
                    distance = closest_distance
                    goto_xy = beacon

            return actions.FUNCTIONS.Move_screen("now", goto_xy)
        else:
            return actions.FUNCTIONS.select_army("select")

agent_class = MoveToBeacon


# In[ ]:


FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'kernel')

app.run(main)

