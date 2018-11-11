#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from multiprocessing import Queue, Pipe, Process
import time

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch
import torch

from sc2ai.actor_critic import ConvActorCritic
from sc2ai.learner import Learner

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("use_cuda", True, "Whether to train on gpu")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")


def run_loop(agents, env, max_frames=0, max_episodes=0):
    """A run loop to have agents and an environment interact."""
    total_frames = 0
    total_episodes = 0
    start_time = time.time()

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(obs_spec, act_spec)

    try:
        while not max_episodes or total_episodes < max_episodes:
            total_episodes += 1
            timesteps = env.reset()
            for a in agents:
                a.reset()
            while True:
                total_frames += 1
                actions = [agent.step(timestep)
                           for agent, timestep in zip(agents, timesteps)]
                if max_frames and total_frames >= max_frames:
                    return
                if timesteps[0].last():
                    break
                timesteps = env.step(actions)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))


def run_thread(agent_classes, players, map_name, visualize, args):
    """Run one thread worth of the environment with agents."""
    with sc2_env.SC2Env(
            map_name=map_name,
            players=players,
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=FLAGS.feature_screen_size,
                feature_minimap=FLAGS.feature_minimap_size,
                rgb_screen=FLAGS.rgb_screen_size,
                rgb_minimap=FLAGS.rgb_minimap_size,
                action_space=FLAGS.action_space,
                use_feature_units=FLAGS.use_feature_units),
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            disable_fog=FLAGS.disable_fog,
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agents = [agent_cls(*args) for agent_cls in agent_classes]
        run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes)
        if FLAGS.save_replay:
            env.save_replay(agent_classes[0].__name__)


def main(unused_argv):
    """Run an agent."""
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    map_inst = maps.get(FLAGS.map)

    agent_classes = []
    players = []

    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    agent_classes.append(agent_cls)

    # players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race], name=FLAGS.agent_name or agent_name))
    players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race]))

    if map_inst.players >= 2:
        if FLAGS.agent2 == "Bot":
            players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                       sc2_env.Difficulty[FLAGS.difficulty]))
        else:
            agent_module, agent_name = FLAGS.agent2.rsplit(".", 1)
            agent_cls = getattr(importlib.import_module(agent_module), agent_name)
            agent_classes.append(agent_cls)
            players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent2_race],
                                         FLAGS.agent2_name or agent_name))

    use_cuda = True
    processes = []

    if not use_cuda:
        state_shape = [2, 84, 84]
        num_actions = 2
        screen_shape = (84, 64)

        device = torch.device('cpu')
        model = ConvActorCritic(num_actions, screen_shape, state_shape).to(device).share_memory()

        for process_id in range(FLAGS.parallel - 1):
            p = Process(target=run_thread, args=(agent_classes, players, FLAGS.map, False, [model, False]))
            p.start()
            processes.append(p)
        run_thread(agent_classes, players, FLAGS.map, False, model)

    else:
        pipes = []
        data_queue = Queue()
        for process_id in range(FLAGS.parallel):
            parent_conn, child_conn = Pipe()
            pipes.append(parent_conn)
            p = Process(target=run_thread, args=(agent_classes, players, FLAGS.map, FLAGS.render,
                                                 [data_queue, child_conn, process_id]))
            processes.append(p)
            p.start()

        Learner(data_queue, pipes)

    for p in processes:
        p.join()

    if FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
