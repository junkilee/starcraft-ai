from pysc2.lib import actions
from pysc2.lib import features

screen_size = 84
minimap_size = 64

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
