import numpy as np
from pysc2.lib import actions as pysc2_actions
from pysc2.lib.features import PlayerRelative, FeatureUnit
from abc import ABC, abstractmethod
from scipy.misc import imsave


class EnvironmentInterface(ABC):
    state_shape = None
    screen_dimensions = None
    num_actions = None

    @classmethod
    def _make_generator(cls, actions):
        yield
        for action in actions:
            yield action
        yield

    @classmethod
    @abstractmethod
    def convert_action(cls, action):
        """
        Converts an action output from the agent into a pysc2 action.
        :return: pysc2 action object
        """
        pass

    @classmethod
    @abstractmethod
    def convert_state(cls, timestep):
        """
        :param timestep: Timestep obtained from pysc2 environment step.
        :return: Tuple of converted state (shape self.state_shape) and action mask
        """
        pass

    @classmethod
    def dummy_state(cls):
        return np.ones(cls.state_shape), np.ones((cls.num_actions,))


class RoachesEnvironmentInterface(EnvironmentInterface):
    state_shape = [3, 84, 84]
    screen_dimensions = [84, 84, 84, 84]
    num_actions = 3

    @classmethod
    def _get_action_mask(cls, timestep):
        mask = np.ones([cls.num_actions])
        if pysc2_actions.FUNCTIONS.Attack_screen.id not in timestep.observation.available_actions:
            mask[0] = 0
        return mask

    @classmethod
    def _get_center_of_mass(cls, obs, alliance=PlayerRelative.SELF):
        """returns the x,y of the center of mass for allied units
        returns None if no allied units are present"""
        num_units = 0
        total_x = 0
        total_y = 0
        for unit in obs.observation.feature_units:
            if unit.alliance == alliance:
                total_x += unit[FeatureUnit.x]
                total_y += unit[FeatureUnit.y]
                num_units += 1
        if num_units != 0:
            return np.array([total_x / num_units, total_y / num_units])
        else:
            return None

    @classmethod
    def _save_timestep_image(cls, timestep, name='outimages/test'):
        imsave(name + '1.png', cls.convert_state(timestep)[0][0])
        imsave(name + '2.png', cls.convert_state(timestep)[0][1])
        imsave(name + '3.png', cls.convert_state(timestep)[0][2])

    @classmethod
    def _rotate_action(cls, x, y):
        yield
        timestep = yield pysc2_actions.FUNCTIONS.select_point('select', (x, y))
        if len(timestep.observation.multi_select) == 0:
            if pysc2_actions.FUNCTIONS.Move_screen.id in timestep.observation.available_actions:
                center_allies = cls._get_center_of_mass(timestep, PlayerRelative.SELF)
                center_enemies = cls._get_center_of_mass(timestep, PlayerRelative.ENEMY)
                if center_enemies is not None and center_allies is not None:
                    direction = center_allies - center_enemies
                    target = [x, y] + 10 * direction
                    target = np.clip(target, a_min=0, a_max=83)
                    yield pysc2_actions.FUNCTIONS.Move_screen('now', target)
        yield

    @classmethod
    def convert_action(cls, action):
        action_index, coords = action
        actions = [
            cls._make_generator([pysc2_actions.FUNCTIONS.Attack_screen('now', coords[0:2])]),
            cls._rotate_action(coords[2], coords[3]),
            cls._make_generator([pysc2_actions.FUNCTIONS.select_army('select')])
        ]
        return actions[action_index]

    @classmethod
    def convert_state(cls, timestep):
        feature_screen = timestep.observation.feature_screen
        beacon = (np.array(feature_screen.player_relative) == PlayerRelative.ENEMY).astype(np.float32)
        player = (np.array(feature_screen.player_relative) == PlayerRelative.SELF).astype(np.float32)
        health = np.array(feature_screen.unit_hit_points).astype(np.float32)
        return np.stack([beacon, player, health], axis=0), cls._get_action_mask(timestep)


class BeaconEnvironmentInterface(EnvironmentInterface):
    state_shape = [2, 84, 84]
    screen_dimensions = [84, 63]
    num_actions = 2

    @classmethod
    def _get_action_mask(cls, timestep):
        mask = np.ones([cls.num_actions])
        if pysc2_actions.FUNCTIONS.Attack_screen.id not in timestep.observation.available_actions:
            mask[0] = 0
        return mask

    @classmethod
    def convert_action(cls, action):
        action_index, coords = action
        actions = [
            cls._make_generator([pysc2_actions.FUNCTIONS.Attack_screen('now', coords)]),
            cls._make_generator([pysc2_actions.FUNCTIONS.select_army('select')])
        ]
        return actions[action_index]

    @classmethod
    def convert_state(cls, timestep):
        player_relative = timestep.observation.feature_screen.player_relative
        beacon = (np.array(player_relative) == 3).astype(np.float32)
        player = (np.array(player_relative) == 1).astype(np.float32)
        return np.stack([beacon, player], axis=0), cls._get_action_mask(timestep)
