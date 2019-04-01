"""A script to train a Ray agent using the :py:mod:`sc2ai` environment.
"""
import os
import sys
from sc2ai.rllib import train
from sc2ai.rllib import rollout

def run():
    parser = argparse.ArgumentParser(
        description="Train or Run an RLlib Agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE)
    subcommand_group = parser.add_subparsers(
        help="Commands to train or run an RLlib agent.", dest="command")

    # see _SubParsersAction.add_parser in
    # https://github.com/python/cpython/blob/master/Lib/argparse.py
    train_parser = train.create_parser(
        lambda **kwargs: subcommand_group.add_parser("train", **kwargs))
    rollout_parser = rollout.create_parser(
        lambda **kwargs: subcommand_group.add_parser("rollout", **kwargs))
    options = parser.parse_args()

    if options.command == "train":
        train.run(options, train_parser)
    elif options.command == "rollout":
        rollout.run(options, rollout_parser)
    else:
        parser.print_help()

if __name__ == '__main__':
    print(os.environ)
    print(sys.argv)
    run()