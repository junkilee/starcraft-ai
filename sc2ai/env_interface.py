import numpy as np
from pysc2.lib import actions as pysc2_actions
from pysc2.lib import static_data
from pysc2.lib.features import PlayerRelative, FeatureUnit, Features
from abc import ABC, abstractmethod


class EnvironmentInterface(ABC):
    state_shape = None
    screen_dimensions = None
    num_actions = None

    @staticmethod
    def _make_generator(actions):
        """
        Utility method to create a generator out of a list of actions. Useful for the case where an action
        does not depend on the state at all. Has yields in the correct position to fit the format needed for the
        learner, but does not do anything with the state information passed to the yield.

        :param actions: List of actions to turn into a generator
        :return: A generator that yields the actions.
        """
        yield
        for action in actions:
            yield action
        yield

    def convert_states(self, timesteps):
        state_input = np.stack([self.convert_state(timestep)[0] for timestep in timesteps], axis=0)
        mask_input = np.stack([self.convert_state(timestep)[1] for timestep in timesteps], axis=0)
        return state_input, mask_input

    @abstractmethod
    def convert_action(self, action):
        """
        Converts an action output from the agent into a pysc2 action.
        :param action: A tuple of (action_index, coordinates)
        :return: A generator of pysc2 action objects
        """
        pass

    @abstractmethod
    def convert_state(self, timestep):
        """
        :param timestep: Timestep obtained from pysc2 environment step.
        :return: Tuple of converted state (shape self.state_shape) and action mask
        """
        pass

    def dummy_state(self):
        return np.ones(self.state_shape), np.ones((self.num_actions,))


class EmbeddingInterfaceWrapper(EnvironmentInterface):
    """
    Wraps an environment interface to add information about unit embeddings to the state. The state becomes a dictionary
    with keys:
        "unit_embeddings": numpy array of shape `[num_units, embedding_size]`, where `num_units` is arbitrary
        "screen": numpy array of shape `[*self.state_shape]`
    """

    def __init__(self, interface):
        self.interface = interface
        self.unit_embedding_size = len(self._get_embedding_columns()) + len(static_data.UNIT_TYPES)

        self.state_shape = self.interface.state_shape
        self.screen_dimensions = self.interface.screen_dimensions
        self.num_actions = self.interface.num_actions

    def convert_action(self, action):
        return self.interface.convert_action(action)

    def convert_state(self, timestep):
        state, action_mask = self.interface.convert_state(timestep)
        return {
            "screen": state,
            "unit_embeddings": self._get_unit_embeddings(timestep)
        }, action_mask

    def dummy_state(self):
        num_units = 12  # TODO: Change back to zero after implementing unit embeddings
        state, action_mask = self.interface.dummy_state()
        return {
            "screen": state,
            "unit_embeddings": np.zeros((num_units, self.unit_embedding_size))
        }, action_mask

    
    def _get_embedding_columns(self):
        return np.array([FeatureUnit.alliance,
                         FeatureUnit.health,
                         FeatureUnit.shield,
                         FeatureUnit.energy,
                         FeatureUnit.owner,
                         FeatureUnit.x,
                         FeatureUnit.y])

    def _get_unit_embeddings(self, timestep):
        unit_info = np.array(timestep.observation.feature_units)
        if unit_info.shape[0] == 0:
            return np.zeros((0, self.unit_embedding_size))
        useful_columns = self._get_embedding_columns()
        adjusted_info = unit_info[:, np.array(useful_columns)]

        num_unit_types = len(static_data.UNIT_TYPES)
        blizzard_unit_type = unit_info[:, FeatureUnit.unit_type]
        pysc2_unit_type = [static_data.UNIT_TYPES.index(t) for t in blizzard_unit_type]
        one_hot_unit_types = np.eye(num_unit_types)[pysc2_unit_type]

        unit_vector = np.concatenate([adjusted_info, one_hot_unit_types], axis=-1)
        return unit_vector


class RoachesEnvironmentInterface(EnvironmentInterface):
    state_shape = [3, 84, 84]
    screen_dimensions = [84, 84, 84, 84, 84, 84]
    num_actions = 4

    @classmethod
    def _get_action_mask(cls, timestep):
        mask = np.ones([cls.num_actions])
        if pysc2_actions.FUNCTIONS.Attack_screen.id not in timestep.observation.available_actions:
            mask[0] = 0
            mask[1] = 0
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
                    if pysc2_actions.FUNCTIONS.Effect_Blink_screen.id in timestep.observation.available_actions:
                        yield pysc2_actions.FUNCTIONS.Effect_Blink_screen('now', target)
                    else:
                        yield pysc2_actions.FUNCTIONS.Move_screen('now', target)
        yield pysc2_actions.FUNCTIONS.select_army('select')
        yield

    @classmethod
    def convert_action(cls, action):
        action_index, coords = action
        coords = coords if coords is not None else (9, 14)
        actions = [
            cls._make_generator([pysc2_actions.FUNCTIONS.Attack_screen('now', coords)]),
            cls._make_generator([pysc2_actions.FUNCTIONS.Move_screen('now', coords)]),
            cls._rotate_action(*coords),
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


class BanelingsEnvironmentInterface(RoachesEnvironmentInterface):
    state_shape = [3, 84, 84]
    screen_dimensions = [84, 84, 84, 84, 84, 84]
    num_actions = 4

    @classmethod
    def _get_action_mask(cls, timestep):
        mask = np.ones([cls.num_actions])
        if pysc2_actions.FUNCTIONS.Attack_screen.id not in timestep.observation.available_actions:
            mask[0] = 0
            mask[1] = 0
        return mask

    @classmethod
    def _sacrifice_action(cls, x, y):
        yield
        timestep = yield pysc2_actions.FUNCTIONS.select_point('select', (x, y))
        if len(timestep.observation.multi_select) == 0:
            if pysc2_actions.FUNCTIONS.Move_screen.id in timestep.observation.available_actions:
                center_allies = cls._get_center_of_mass(timestep, PlayerRelative.SELF)
                center_enemies = cls._get_center_of_mass(timestep, PlayerRelative.ENEMY)
                if center_enemies is not None and center_allies is not None:
                    direction = center_allies - center_enemies
                    target = [x, y] - 10 * direction
                    target = np.clip(target, a_min=0, a_max=83)
                    yield pysc2_actions.FUNCTIONS.Move_screen('now', target)
        yield

    @classmethod
    def convert_action(cls, action):
        action_index, coords = action
        coords = coords if coords is not None else (9, 14)
        actions = [
            cls._make_generator([pysc2_actions.FUNCTIONS.Attack_screen('now', coords)]),
            cls._make_generator([pysc2_actions.FUNCTIONS.Move_screen('now', coords)]),
            cls._sacrifice_action(*coords),
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
    screen_dimensions = [84, 84]
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
        coords = coords if coords is not None else (9, 14)
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
