import argparse
from baselines.common.misc_util import ( boolean_flag )
from pysc2.env import sc2_env

def parse_args():
    parser = argparse.ArgumentParser("Starcraft Environment Test")
    parser = add_default_sc2_arguments(parser)
    return parser.parser_args()

def add_default_sc2_arguments(parser):
    boolean_flag(parser, "render", type=True, help="Whether to render with pygame.")
    parser.add_argument("feature_screen_size", type=int, default=84, help="Resolution for screen feature layers.")
    parser.add_argument("feature_minimap_size", type=int, default=64, help="Resolution for screen feature layers.")
    parser.add_argument("rgb_screen_size", type=int, default=None, help="Resolution for rendered screen.")
    parser.add_argument("rgb_minimap_size", type=int, default=None, help="Resolution for rendered minimap.")

flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_, "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
# newly added (BROWN MOD)
flags.DEFINE_bool("use_raw_units", True,
                  "Whether to include raw units.")
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
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")


