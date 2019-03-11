from pysc2.lib import actions
from pysc2.lib import features

screen_size = 84
minimap_size = 64

DEFAULT_ENV_OPTIONS = {
    "render": False,
    "feature_screen_size": 84,
    "feature_minimap_size: 64,
    "rgb_screen_size": None,
    "rgb_minimap_size": None,
    "action_space": sc2_env.ActionSpace.FEATURES, # RGB
    "use_feature_units": False,
    "use_raw_units": True,
    "disable_fog": True,
    "max_agent_steps": 0,
    "game_steps_per_episode": None,
    "max_episodes": 0,
    "step_mul":80,
    "agent": "pysc2.agents.random_agent.RandomAgent",
    "agent_name": None,
    "agent_race": sc2_env.Race.terran,
    "agent2": "Bot",
    "agent2_name": None,
    "agent2_race": sc2_env.Race.terran,
    "difficulty": sc2_env.Difficulty.very_easy,
    "profile": False,
    "trace": False,
    "parallel": 1,
    "save_replay": True,
    "map": None
}

class ActionIDs:
    NO_OP = actions.FUNCTIONS.no_op.id

def get_sc2_full_action_list(filter = []):
    feats = features.Features(
      features.AgentInterfaceFormat(
      feature_dimensions=features.Dimensions(
	  screen=FLAGS.screen_size,
	  minimap=FLAGS.minimap_size)))
    action_spec = feats.action_spec()
    flattened = 0
    count = 0
    for func in action_spec.functions:
    if FLAGS.hide_specific and actions.FUNCTIONS[func.id].general_id != 0:
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
