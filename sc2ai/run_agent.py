#!/usr/bin/python# Copyright 2017 Google Inc. All Rights Reserved.
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

from multiprocessing import Pipe, Process
import numpy as np

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.env.environment import StepType

from sc2ai.learner import Learner
from pysc2.lib import actions as pysc2_actions

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


class RoachesEnvironmentInterface:
    """
    Facilitates communication between the agent and the environment.
    """

    def __init__(self):
        self.state_shape = [2, 84, 84]
        self.screen_shape = [84, 63]
        self.num_actions = 2

    def _get_action_mask(self, timestep):
        mask = np.ones([self.num_actions])
        if pysc2_actions.FUNCTIONS.Attack_screen.id not in timestep.observation.available_actions:
            mask[0] = 0
        return mask

    def convert_action(self, *action):
        """
        Converts an action output from the agent into a pysc2 action.
        :return: pysc2 action object
        """
        action_index, x, y = action
        if action_index == 0:
            return pysc2_actions.FUNCTIONS.Attack_screen('now', (x, y))
        else:
            return pysc2_actions.FUNCTIONS.select_army('select')

    def convert_state(self, timestep):
        """
        :param timestep: Timestep obtained from pysc2 environment step.
        :return: Tuple of converted state (shape self.state_shape) and action mask
        """
        player_relative = timestep.observation.feature_screen.player_relative
        beacon = (np.array(player_relative) == 3).astype(np.float32)
        player = (np.array(player_relative) == 1).astype(np.float32)
        return np.stack([beacon, player], axis=0), self._get_action_mask(timestep)


class SCEnvironmentWrapper:
    def __init__(self, map_name, agent_interface, visualize):
        self.env = sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race[FLAGS.agent_race])],
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
            visualize=visualize)
        self.env.__enter__()
        self.curr_timestep = None
        self.agent_interface = agent_interface

    # TODO: Check the right timestep is being sent in
    def step(self, action):
        timestep = self.curr_timestep
        self.curr_timestep = self.env.step([self.agent_interface.convert_action(*action)])[0]
        reward = self.curr_timestep.reward
        done = int(self.curr_timestep.step_type == StepType.LAST)
        state, action_mask = self.agent_interface.convert_state(timestep)
        return state, action_mask, reward, done

    def reset(self):
        self.curr_timestep = self.env.reset()[0]
        state, action_mask = self.agent_interface.convert_state(self.curr_timestep)
        return state, action_mask,  0, int(False)

    def close(self):
        self.env.__exit__(None, None, None)


def run_process(env_factory, pipe):
    environment = env_factory()
    while True:
        endpoint, data = pipe.recv()

        if endpoint == 'step':
            pipe.send(environment.step(data))
        elif endpoint == 'reset':
            pipe.send(environment.reset())
        elif endpoint == 'close':
            environment.close()
            pipe.close()
        else:
            raise Exception("Unsupported endpoint")


class MultipleEnvironment:
    def __init__(self, env_factory, num_instance=1):
        self.pipes = []
        self.processes = []
        self.num_instances = num_instance
        for process_id in range(num_instance):
            parent_conn, child_conn = Pipe()
            self.pipes.append(parent_conn)
            p = Process(target=run_process, args=(env_factory, child_conn))
            self.processes.append(p)
            p.start()

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))
        return self.get_results()

    def reset(self):
        for pipe in self.pipes:
            pipe.send(('reset', None))
        return self.get_results()

    def get_results(self):
        states, masks, rewards, dones = zip(*[pipe.recv() for pipe in self.pipes])
        return np.stack(states), np.stack(masks), np.stack(rewards), np.stack(dones)

    def close(self):
        for pipe in self.pipes:
            pipe.send(('close', None))
        for process in self.processes:
            process.join()


def main(unused_argv):
    interface = RoachesEnvironmentInterface()
    environment = MultipleEnvironment(lambda: SCEnvironmentWrapper(FLAGS.map, interface, visualize=False),
                                      num_instance=FLAGS.parallel)
    learner = Learner(environment, interface, use_cuda=True)

    try:
        for i in range(1000):
            learner.train_episode()
    finally:
        environment.close()


if __name__ == "__main__":
    app.run(main)
