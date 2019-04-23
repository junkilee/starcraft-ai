from pysc2.lib import actions as pysc2_actions
from abc import ABC, abstractmethod

class ActionList():
    """

    """
    def __init__(self, actionlist):
        pass

    def convert_to_gym_action_spaces(actionlist):
        registered

        for action in actionlist:
            pass

        gym_space = None

        return gym_space


class Action(ABC):
    """
    An abstract class for a default action which relies only on the policy network's output

    """
    def __init__(self, function_type, **kwargs):
        self.function_type = function_type
        self.default_arguments = kwargs

    def __call__(self, action_values):

        return

    @abstractmethod
    def parameter_type(self):
        pass

class HighLevelAction(Action):
    """
    An abstract class for a high level action which relies on an agent's observation other than the policy network's
    output itself

    """
    def __init__(self, function_type, **kwargs):
        super().__init__(function_type, **kwargs)

    def __call__(self, action_values):

        return

    @abstractmethod
    def parameter_type(self):
        pass


def SelectAction(Action):
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


