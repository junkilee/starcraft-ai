#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle

import sc2ai.envs as sc2ai_env
from absl import flags
from sc2ai.spinup.utils.mpi_pytorch import num_procs, proc_id
import numpy as np


EXAMPLE_USAGE = """
Example Usage via run.py:
    python3 run.py envtest MoveToBeacon 

Example Usage via executable:
    python3 ./rllib/envtest.py MoveToBeacon
"""


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Test the environment settings associated with the given minimap.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "mapname", type=str, help="Name of the minimap.")
    #parser.add_argument(
    #    "out", type=str, help="Name of the minimap.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
            "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--realtime",
        default=False,
        action="store_const",
        const=True,
        help="Determine whether the game will run in realtime.")
    parser.add_argument(
        "--skips", default=10, help="Number of action skips before a random action.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps for testing the environment.")
    return parser


def run(args, parser):
    num_steps = int(args.steps)
    envtest(args.mapname, num_steps, args.out, args.no_render, args.realtime, int(args.skips))


def envtest(mapname, num_steps, out=None, no_render=True, realtime=False, num_skips=0):
    env = sc2ai_env.make_sc2env(map=mapname, render=not no_render, realtime=realtime)

    env._action_set.report()
    print(env.action_gym_space)
    print(env.action_set.get_action_spec())
    print(env.action_gym_space.nvec)
    print(env.observation_gym_space)
    print("mpi proc : {}/{}".format(proc_id(), num_procs()))
    max_steps_per_episode = 10000
    total_steps = 0
    while total_steps < (num_steps or total_steps + 1):
        if out is not None:
            rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        steps = 0
        while not done and steps < max_steps_per_episode:
            # randomly sample an action
            actions = [env.sample_action()]
            next_state, reward, done, _ = env.step(actions)
            print("=>", actions, steps, reward, done)
            print(next_state['feature_screen'].shape)
            reward_total += reward
            if not no_render:
                env.render()
            if out is not None:
                rollout.append([state, actions, next_state, reward, done])
            steps += 1
            total_steps += 1
            state = next_state
        if out is not None:
            rollout.append(rollout)
        print("{} steps and episode reward {}".format(steps, reward_total))
    if out is not None:
        pickle.dump(rollout, open(out, "wb"))


if __name__ == "__main__":
    flags.FLAGS.mark_as_parsed()
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
