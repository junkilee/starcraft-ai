from absl import app
from absl import flags
import os
import pathlib

from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.maps import lib

import sc2ai.env_interface as interfaces
from sc2ai.environment import MultipleEnvironment, SCEnvironmentWrapper
from sc2ai.tflearner.tflearner import ActorCriticLearner
from sc2ai.tflearner.tf_agent import InterfaceAgent, ConvAgent, LSTMAgent

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

def log_run_config():
    # Log all flags to {save_dir}/info.txt

    pathlib.Path(_save_dir()).mkdir(parents=True, exist_ok=True)
    run_info_path = os.path.join(_save_dir(), 'info.txt')

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

# ------------------------ GET FLAGS ------------------------
def _save_dir():
    return os.path.join('saves', FLAGS.save_name)

def _num_parallel_instances():
    return 1 if FLAGS.render else FLAGS.parallel

# ------------------------ FACTORIES ------------------------

def build_env_kwargs():
    return {
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
        'replay_dir': os.path.join(_save_dir(), 'replays')
    }

def build_agent_interface(map_name):
    if map_name in {'DefeatRoaches', 'StalkersVsRoaches'}:
        return interfaces.EmbeddingInterfaceWrapper(interfaces.RoachesEnvironmentInterface())
    elif map_name == 'MoveToBeacon':
        return interfaces.EmbeddingInterfaceWrapper(interfaces.BeaconEnvironmentInterface())
    elif map_name == 'DefeatZerglingsAndBanelings':
        return interfaces.BanelingsEnvironmentInterface()
    elif map_name == 'BuildMarines':
        return interfaces.EmbeddingInterfaceWrapper(interfaces.TrainMarines())
    else:
        raise Exception('Unsupported Map')

# ------------------------ RUNNER CLASS ------------------------

class AgentRunner:
    # this class should not rely at all on the global FLAGS

    def __init__(self, sc2map, hyperparams):
        self.hyperparams = hyperparams
        self.sc2map = sc2map

        self.initialize()

    def initialize(self, reset=False):
        # Initializes interface, env, agent and learner. On reset we load the model
        self.agent_interface = build_agent_interface(self.sc2map)

        self.environment = MultipleEnvironment(lambda: SCEnvironmentWrapper(self.agent_interface, build_env_kwargs()),
                                          num_instance=_num_parallel_instances())
        
        self.agent = LSTMAgent(self.agent_interface)

        # On resets, always load the model
        load_model = True if reset else self.hyperparams['load_model']

        self.learner = ActorCriticLearner(self.environment, self.agent,
                                     save_dir=_save_dir(),
                                     load_model=load_model,
                                     gamma=self.hyperparams['gamma'],
                                     td_lambda=self.hyperparams['td_lambda'],
                                     learning_rate=self.hyperparams['learning_rate'])

    
    def train_agent(self, num_training_episodes, reset_env_every=1000):
        # Trains the agent for num_training_episodes
        # Resets the environment every once in a while (to deal with memory leak)

        self.episode_count = 0
        self.reset_env_every = reset_env_every
    
        print("Runner: beinning training for", num_training_episodes, "episodes")
        while self.episode_count < num_training_episodes:

            if self._should_reset():
                print("Runner: Decided to reset")
                self._reset()

            self._train_episode()
        
        print("Runner: training fininshed, trained", self.episode_count, "episodes")
        self._close_env()


    def _should_reset(self):
        # Based on episode count only, not crashes
        return self.episode_count % self.reset_env_every == 0 and self.episode_count > 0


    def _train_episode(self):
        try:
            self.learner.train_episode()
        except:
            print("Runner: Encountered error in train_episode")
            self._reset()

    
    def _reset(self):
        # Shutdown env and reinitialize everything
        print("Runner: Resetting")
        self._close_env()
        self.initialize(reset=True)

    
    def _close_env(self):
        # Tell env to shutdown
        print("Runner: Shutting down environment")
        self.environment.close()


# ------------------------ MAIN ------------------------

def main(unused_argv):
    # Log the configuration for this run
    log_run_config()

    hyperparameters = {
        'load_model': FLAGS.load_model,
        'gamma': FLAGS.gamma,
        'td_lambda': FLAGS.td_lambda,
        'learning_rate': FLAGS.learning_rate
    }

    # Build Runner
    runner = AgentRunner(sc2map=FLAGS.map, hyperparams=hyperparameters)

    # Train the agent
    runner.train_agent(FLAGS.max_episodes)


if __name__ == "__main__":
    app.run(main)
