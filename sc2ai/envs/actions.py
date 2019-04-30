from pysc2.lib import actions
from abc import ABC, abstractmethod

class ActionList():
    """Stores a action list for an environment."""
    def __init__(self, actions_list):
        self.actions_list = actions_list

        for i, action in enumerate(self.actions_list):
            action.register_action_id(i)

        self.num_actions = len(actions_list)
        self.current_num_actions = self.num_actions

    @classmethod
    def add_all_sc2_actions(cls):
        raise NotImplementedError()

    def update_available_actions(self, available_actions):



    def convert_to_gym_action_spaces(self, actionlist):
        vectors = []

        vectors.add(self.num_actions)

        for action in actionlist:
            pass

        gym_space = None

        return gym_space


class Action(ABC):
    """An abstract class for a default action which relies only on the policy network's output"""
    def __init__(self, function_type, arg_types, **kwargs):
        self.function_type = function_type
        self.default_arguments = kwargs
        self.arg_types = arg_types
        self.action_id = None



    @abstractmethod
    def parameter_type(self):
        pass

class HighLevelAction(Action):
    """
    An abstract class for a high level action which relies on an agent's observation other than the policy network's
    output itself

    """
    def __init__(self, function_type, arg_types, **kwargs):
        super().__init__(function_type, arg_types, **kwargs)

    def __call__(self, action_values):

        return

    @abstractmethod
    def parameter_type(self):
        pass


def SelectAction(Action):
    def __init__(self, function_type, **kwargs):
        super().__init__(function_type, **kwargs)
        self.arg_types = []

    def __call__(self):
        return

    def parameter_type(self):
        return None

def SelectAllAction(Action):
    def __call__(self):
        return

    def parameter_type(self):
        return None


def MoveAction(Action):
    def __call__(self):
        return

    def parameter_type(self):
        return None

def AttackAction(Action):
    def __call__(self):
        return

    def parameter_type(self):
        return None


