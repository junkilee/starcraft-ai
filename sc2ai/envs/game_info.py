from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from collections import namedtuple

screen_size = 84
minimap_size = 64


SC2EnvOptions = namedtuple('SC2EnvOptions', ('render', 'feature_screen_size', 'feature_minimap_size',
                                             'rgb_screen_size', 'rgb_minimap_size', 'action_space',
                                             'use_feature_units', 'use_raw_units', 'disable_fog',
                                             'max_agent_steps', 'game_steps_per_episode', 'max_episodes',
                                             'step_mul',
                                             'agent1_name', 'agent1_race', 'agent2_name', 'agent2_race',
                                             'difficulty', 'profile', 'trace', 'parallel', 'save_replay'))
""" The definition of SC2EnvOptions """

default_env_options = SC2EnvOptions(render=False,
                                    feature_screen_size=screen_size,
                                    feature_minimap_size=minimap_size,
                                    rgb_screen_size=None,
                                    rgb_minimap_size=None,
                                    action_space=sc2_env.ActionSpace.FEATURES,
                                    use_feature_units=False,
                                    use_raw_units=True,
                                    disable_fog=True,
                                    max_agent_steps=0,
                                    game_steps_per_episode=None,
                                    max_episodes=0,
                                    step_mul=80,
                                    agent1_name="TrainedAI",
                                    agent1_race=sc2_env.Race.terran,
                                    agent2_name="DefaultAI",
                                    agent2_race=sc2_env.Race.terran,
                                    difficulty=sc2_env.Difficulty.very_easy,
                                    profile=False,
                                    trace=False,
                                    parallel=1,
                                    save_replay=True)
""" The default value for the SC2EnvOptions. """

class ActionIDs:
    NO_OP = actions.FUNCTIONS.no_op.id

def get_sc2_full_action_list(hide_specific = False):
    feats = features.Features(
      features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(
            screen=screen_size,
            minimap=minimap_size)))
    action_spec = feats.action_spec()
    flattened = 0
    count = 0
    for func in action_spec.functions:
        if hide_specific and actions.FUNCTIONS[func.id].general_id != 0:
            continue
        count += 1
        act_flat = 1
        for arg in func.args:
            for size in arg.sizes:
                act_flat *= size
                flattened += act_flat
    #print(func.str(True))
    #print("Total base actions:", count)
    #print("Total possible actions (flattened):", flattened)
