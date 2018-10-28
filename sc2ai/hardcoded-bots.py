# Agents are implemented here with the help of below references.
# https://github.com/deepmind/pysc2/tree/master/pysc2/agents
# https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c
from PIL import Image
import numpy as np


from pysc2.agents import base_agent
# from pysc2.agents.base_agent import _xy_locs
from pysc2.lib import actions
from pysc2.lib import units
import time
from pysc2.lib import features


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


class DefeatZerglingsAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(DefeatZerglingsAgent, self).step(obs)

        if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            enemies = _xy_locs(player_relative == features.PlayerRelative.ENEMY)
            if not enemies:
                return actions.FUNCTIONS.no_op()

            target = enemies[np.argmax(np.array(enemies)[:, 1])]
            return actions.FUNCTIONS.Attack_screen('now', target)

        if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
            return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.no_op()


class FindAndDefeatZerglingsAgent(base_agent.BaseAgent):
    steps_until_change_destination = 0
    random_move_duration = 6

    def step(self, obs):
        super(FindAndDefeatZerglingsAgent, self).step(obs)

        if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            enemies = _xy_locs(player_relative == features.PlayerRelative.ENEMY)

            # If there are no enemies in range, move to a random location for n frames
            if not enemies:
                if self.steps_until_change_destination == 0:
                    rand_x = np.random.randint(1, player_relative.shape[0])
                    rand_y = np.random.randint(1, player_relative.shape[1])
                    destination = [rand_x, rand_y]
                    self.steps_until_change_destination = self.random_move_duration
                    return actions.FUNCTIONS.Attack_screen('now', destination)
                self.steps_until_change_destination -= 1
                return actions.FUNCTIONS.no_op()
            target = enemies[np.argmax(np.array(enemies)[:, 1])]
            return actions.FUNCTIONS.Attack_screen('now', target)

        if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
            return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.no_op()


class CollectMineralsAndGas(base_agent.BaseAgent):
    curr_frame_num = 0
    harvesting = False
    selected_single = False

    def reset(self):
        super(CollectMineralsAndGas, self).reset()
        self.harvesting = False

    def step(self, obs):
        self.curr_frame_num += 1

        super(CollectMineralsAndGas, self).step(obs)

        if not self.harvesting:
            if actions.FUNCTIONS.Harvest_Gather_screen.id in obs.observation.available_actions:
                unit_types = obs.observation.feature_screen.unit_type
                coords = _xy_locs(unit_types == units.Neutral.MineralField)
                target = coords[0]
                target = [target[0] + 2, target[1] + 2]
                print(target)

                self.harvesting = True
                return actions.FUNCTIONS.Harvest_Gather_screen('now', target)

            if actions.FUNCTIONS.select_rect.id in obs.observation.available_actions:
                return actions.FUNCTIONS.select_rect('select', [1, 1], [83, 83])
        else:
            if not self.selected_single:
                unit_types = obs.observation.feature_screen.unit_type
                coords = _xy_locs(unit_types == units.Terran.SCV)
                target = coords[0]
                target = [target[0] + 1, target[1] + 1]
                self.selected_single = True
                return actions.FUNCTIONS.select_point('select', target)
            else:
                if actions.FUNCTIONS.Build_Refinery_screen.id in obs.observation.available_actions:
                    unit_types = obs.observation.feature_screen.unit_type
                    coords = _xy_locs(unit_types == units.Neutral.VespeneGeyser)
                    target = coords[0]
                    target = [target[0] + 2, target[1] + 2]
                    return actions.FUNCTIONS.Build_Refinery_screen('now', target)
        return actions.FUNCTIONS.no_op()












