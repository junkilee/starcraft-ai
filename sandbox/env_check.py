import importlib
import threading

from future.builtins import range  # pylint: disable=redefined-builtin

from absl import app

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

def main(unused_argvs):
  map_name = "MoveToBeacon"
  map_inst = maps.get(map_name)

  players = []

  players.append(sc2_env.Agent(sc2_env.Race["random"], "TestAgent"))

  with sc2_env.SC2Env(
      map_name=map_name,
      players=players,
      agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=84,
          feature_minimap=64,
          rgb_screen=None,
          rgb_minimap=None,
          action_space=None,
          use_feature_units=True,
          use_raw_units=True),
      step_mul=8,
      game_steps_per_episode=None,
      disable_fog=False,
      visualize=False) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    for function in env.action_spec()[0].functions:
        print(function)
    print(env.action_spec()[0].types)
    print(env.observation_spec()[0])

app.run(main)
