import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env.environment import StepType


class ParallelAgent(base_agent.BaseAgent):
    """
    Parallel Agent used to train batched A2C on multiple processors.
    """

    def __init__(self, data_queue, pipe, process_id):
        super().__init__()
        self.data_queue = data_queue
        self.pipe = pipe
        self.process_id = process_id
        self.num_actions = 2

    def get_action_mask(self, available_actions):
        """
        Creates a mask array based on which actions are available

        :param available_actions: List of available action id's provided by pysc2
        :return: A 1 dimensional mask with 1 if an action is available and 0 if not.
        """
        mask = np.ones([self.num_actions])
        if actions.FUNCTIONS.Attack_screen.id not in available_actions:
            mask[0] = 0
        return mask

    def step(self, obs):
        """
        This function is called at each time step. At each step, we collect the (state, action, reward) tuple and
        save it for training.

        :param obs: sc2 observation object
        :return: states, reward, done
        """
        super().step(obs)

        player_relative = obs.observation.feature_screen.player_relative
        beacon = (np.array(player_relative) == 3).astype(np.float32)
        player = (np.array(player_relative) == 1).astype(np.float32)

        action_mask = self.get_action_mask(obs.observation.available_actions)

        state = np.stack([beacon, player], axis=0)

        move_id = np.random.randint(0, 100000)
        data = (state, obs.reward, action_mask, obs.step_type, self.process_id, move_id)

        self.pipe.send(data)
        action_index, x, y, server_move_id = self.pipe.recv()
        assert(server_move_id == move_id)

        if obs.step_type != StepType.LAST:
            if action_index == 0:
                return actions.FUNCTIONS.Attack_screen('now', (x, y))
            else:
                return actions.FUNCTIONS.select_army('select')
