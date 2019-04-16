from absl import app
from absl import flags
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.maps import lib

from sc2ai.runner import AgentRunner

# ------------------------ DEFINE FLAGS ------------------------

if __name__ == '__main__':
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
    flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")
    flags.DEFINE_bool("epsilon", False, "Whether to use epsilon greedy")

    flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
    flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
    flags.DEFINE_integer("max_episodes", 10000, "Total episodes.")
    flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

    flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                      "Agent 1's race.")
    flags.DEFINE_bool("cuda", True, "Whether to train on gpu")
    flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

    flags.DEFINE_string("map", None, "Name of a map to use.")
    flags.DEFINE_string("save_name", "recent", "Save run information under ./saves/<save_name>")

    flags.DEFINE_bool("load_model", False, "Whether to load the previous run's model")

    flags.DEFINE_float("gamma", 0.96, "Discount factor")
    flags.DEFINE_float("learning_rate", 0.0003, "Learning rate")
    flags.DEFINE_float("td_lambda", 0.96, "Lambda value for generalized advantage estimation")

    flags.mark_flag_as_required("map")


def log_run_config(save_dir):
    # Log all flags to {save_dir}/info.txt
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    run_info_path = os.path.join(save_dir, 'info.txt')

    with open(run_info_path, 'a') as f:
        for key in FLAGS.__flags:
            f.write("%s, %s\n" % (key, FLAGS.__flags[key]._value))
        f.write('\n\n\n')

# ------------------------ DEFINE CUSTOM MAPS ------------------------

class StalkersVsRoachesMap(lib.Map):
    directory = "mini_games"
    download = "https://github.com/deepmind/pysc2#get-the-maps"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8

# name = 'StalkersVsRoaches'
# globals()[name] = type(name, (StalkersVsRoachesMap,), dict(filename=name))

# ------------------------ MAIN ------------------------

def main(unused_argv):

    save_dir = os.path.join('saves', FLAGS.save_name)

    runner_params = {
        'save_dir': save_dir,
        'num_parallel_instances': 1 if FLAGS.render else FLAGS.parallel
    }

    model_params = {
        'load_model': FLAGS.load_model,
        'gamma': FLAGS.gamma,
        'td_lambda': FLAGS.td_lambda,
        'learning_rate': FLAGS.learning_rate,
        'save_every': 2
    }

    env_kwargs = {
        'map_name': FLAGS.map,
        'players': [sc2_env.Agent(sc2_env.Race[FLAGS.agent_race])],
        'agent_interface_format': sc2_env.parse_agent_interface_format(
            feature_screen=FLAGS.feature_screen_size,
            feature_minimap=FLAGS.feature_minimap_size,
            rgb_screen=FLAGS.rgb_screen_size,
            rgb_minimap=FLAGS.rgb_minimap_size,
            action_space=FLAGS.action_space,
            use_feature_units=True),
        'step_mul': FLAGS.step_mul,
        'game_steps_per_episode': FLAGS.game_steps_per_episode,
        'disable_fog': FLAGS.disable_fog,
        'visualize': FLAGS.render,
        'save_replay_episodes': 200,
        'replay_dir': os.path.join(save_dir, 'replays')
    }

    # Log the configuration for this run
    log_run_config(save_dir)

    # Build Runner
    runner = AgentRunner(FLAGS.map, runner_params, model_params, env_kwargs)

    # Train the agent
    runner.train_agent(FLAGS.max_episodes)


if __name__ == "__main__":
    app.run(main)
