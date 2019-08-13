#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle

import ray
import sc2ai.envs as sc2ai_env
from absl import flags


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
        "--skips", default=10, help="Number of action skips before a random action.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps for testing the environment.")
    return parser


def run(args, parser):
    ray.init()
    num_steps = int(args.steps)
    envtest(args.mapname, num_steps, args.out, args.no_render, int(args.skips))


def envtest(mapname, num_steps, out=None, no_render=True, num_skips=0):
    env = sc2ai_env.make_sc2env(map=mapname, render=not no_render)

    steps = 0
    while steps < (num_steps or steps + 1):
        if out is not None:
            rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            # randomly sample an action
            actions = [env.sample_action()]
            next_state, reward, done, _ = env.step(actions)
            reward_total += reward
            if not no_render:
                env.render()
            if out is not None:
                rollout.append([state, actions, next_state, reward, done])
            steps += 1
            state = next_state
        if out is not None:
            rollout.append(rollout)
        print("Episode reward", reward_total)
    if out is not None:
        pickle.dump(rollout, open(out, "wb"))


if __name__ == "__main__":
    flags.FLAGS.mark_as_parsed()
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
