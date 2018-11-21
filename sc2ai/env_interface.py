import numpy as np
from pysc2.lib import actions as pysc2_actions


class RoachesEnvironmentInterface:
    """
    Facilitates communication between the agent and the environment.
    """
    state_shape = [2, 84, 84]
    screen_shape = [84, 63]
    num_actions = 2

    @classmethod
    def _get_action_mask(cls, timestep):
        mask = np.ones([cls.num_actions])
        if pysc2_actions.FUNCTIONS.Attack_screen.id not in timestep.observation.available_actions:
            mask[0] = 0
        return mask

    @classmethod
    def convert_action(cls, *action):
        """
        Converts an action output from the agent into a pysc2 action.
        :return: pysc2 action object
        """
        action_index, x, y = action
        if action_index == 0:
            return pysc2_actions.FUNCTIONS.Attack_screen('now', (x, y))
        else:
            return pysc2_actions.FUNCTIONS.select_army('select')

    @classmethod
    def convert_state(cls, timestep):
        """
        :param timestep: Timestep obtained from pysc2 environment step.
        :return: Tuple of converted state (shape self.state_shape) and action mask
        """
        player_relative = timestep.observation.feature_screen.player_relative
        beacon = (np.array(player_relative) == 3).astype(np.float32)
        player = (np.array(player_relative) == 1).astype(np.float32)
        return np.stack([beacon, player], axis=0), cls._get_action_mask(timestep)

    @classmethod
    def dummy_state(cls):
        return np.ones(cls.state_shape)

    @classmethod
    def dummy_mask(cls):
        return np.ones((cls.num_actions,))