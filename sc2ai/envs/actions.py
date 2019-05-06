from pysc2.lib import actions
from sc2ai.envs import game_info
from abc import ABC, abstractmethod

class ActionList:
    """Stores a action list for an environment."""
    def __init__(self, actions_list, reorder_action_id = False):
        self.actions_list = actions_list
        self.reorder_action_id = reorder_action_id

        self.num_actions = len(actions_list)
        self.current_num_actions = self.num_actions

    @classmethod
    def add_all_sc2_actions(cls):
        raise NotImplementedError()

    def update_available_actions(self, available_actions):

    def transform_action(self, observation, action_values):
        action_id = action_values[0]

        if self.reorder_action_id:
            raise NotImplementedError
        else:
            actions = self.actions_list[action_id].transform_action(observation, action_values[1:])

        return actions

    def convert_to_gym_action_spaces(self, actionlist):
        vectors = []

        vectors.add(self.num_actions)

        for action in actionlist:
            vectors.append(action.retrieve_parameter_size_vectors())

        gym_space = None

        return gym_space


class Action(ABC):
    """An abstract class for a default action which relies only on the policy network's output"""
    def __init__(self, arg_types, **kwargs):
        self.default_arguments = kwargs
        self.arg_types = arg_types
        self.action_id = None
        self.feature_screen_size = kwargs['feature_screen_size'] if 'feature_screen_size' in kwargs else None
        self.feature_minimap_size = kwargs['feature_minimap_size'] if 'feature_minimap_size' in kwargs else None

    def retrieve_parameter_size_vectors(self):
        vectors = []
        for arg_type in self.arg_types:
            if arg_type.sizes == (0,0):
                if arg_type is actions.TYPES.screen:
                    feature_screen_size = self.feature_screen_size or game_info.feature_screen_size
                    vectors.add(feature_screen_size**2)
                elif arg_type is actions.TYPES.minimap:
                    feature_minimap_size = self.feature_minimap_size or game_info.feature_minimap_size
                    vectors.add(feature_minimap_size**2)
                else:
                    raise NotImplementedError
            else:
                vectors.add(arg_type.sizes)
        return vectors

    @abstractmethod
    def transform_action(self, observation, action_values):
        pass


class AtomAction(Action):
    """A Class made to directly mirror pysc2 523 default actions"""
    def __init__(self, function_tuple, **kwargs):
        self.function_tuple= function_tuple
        self.function_type = function_tuple.function_type
        arg_types = function_tuple.args
        super().__init__(self, arg_types, **kwargs)

    def transform_action(self):





class SelectPointAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.select_point, **kwargs)


class SelectRectAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.select_point, **kwargs)


class CompoundAction(Action):