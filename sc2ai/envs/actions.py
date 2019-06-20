from abc import ABC, abstractmethod
from pysc2.lib import actions
from sc2ai.envs import game_info
from gym.spaces.multi_discrete import MultiDiscrete
import logging

logger = logging.getLogger(__name__)


def retrieve_parameter_size_vector(arg_type, feature_screen_size, feature_minimap_size):
    if arg_type.sizes == (0, 0):
        if arg_type is actions.TYPES.screen:
            return feature_screen_size ** 2
        elif arg_type is actions.TYPES.minimap:
            return feature_minimap_size ** 2
        else:
            raise NotImplementedError
    else:
        return arg_type.sizes


class ActionSet(ABC):
    """An abstraction class for Action Sets.
    An Action Set handles a set of actions assigned to a specific environment and
    determines which action portfolio it needs to represent to the network.
    The structure of action output is determined by implementing 'convert_to_gym_action_spaces'
    method.
    """

    def __init__(self, action_list):
        self._action_list = action_list
        self._num_actions = len(action_list)
        self._current_available_actions = [True] * self._num_actions

    @abstractmethod
    def convert_to_gym_action_spaces(self):
        pass

    @abstractmethod
    def transform_action(self, observation, action_values):
        pass

    def update_available_actions(self, available_actions):
        """Update the table of available actions

        Args:
            available_actions: a numpy array containing a list of available actions in IDs
            given by PySC2.

        Returns:
            None
        """
        for i, action in enumerate(self._action_list):
            ids = action.get_pysc2_action_ids()
            check = False
            for id in ids:
                if not id in available_actions:
                    check = True
                    break
            if check:
                self._current_available_actions[i] = False


class DefaultActionSet(ActionSet):
    """Store a list of default PySC2 actions in this set.
    Uses a shared parameter space called _parameter_registry."""

    def __init__(self, actionlist, reorder_action_id = False,
                 feature_screen_size = game_info.feature_screen_size,
                 feature_minimap_size = game_info.feature_minimap_size,):
        super().__init__(actionlist)
        self._argument_types_registry = self.register_argument_types()
        self._reorder_action_id = reorder_action_id
        self._current_num_actions = self._num_actions
        self._parameter_registry = self.register_argument_types()
        self._feature_screen_size = feature_screen_size
        self._feature_minimap_size = feature_minimap_size
        self._no_op_action = NoOpAction()

    def register_argument_types(self):
        registry = {}
        count = 0
        for action in self._action_list:
            parameters = action.arg_types
            for parameter in parameters:
                if not parameter in registry:
                    registry[parameter] = count
                    count += 1
        return registry

    @classmethod
    def add_all_basic_sc2_actions(cls):
        """A method to create a DefaultActionSet with all the PySC2 actions.

        Returns:
            A DefaultActionSet containing all the available action set in PySC2.
        """
        action_list = []
        for func in actions.FUNCTIONS:
            cls_ = AtomAction.factory(func)
            globals()[cls_.__name__] = cls_
            action_list.append(cls_)
        return cls(action_list)

    def transform_action(self, observation, action_values):
        action_id = action_values[0]

        if self._reorder_action_id:
            raise NotImplementedError()
        else:
            if action_id < 0 or action_id > len(self._action_list):
                logger.error("The wrong action ID %d from the network output", action_id)
                raise Exception("The wrong action ID. {}".format(action_id))
            #if not self._current_available_actions[action_id]:

            action = self._action_list[action_id]
            parameter_types = action.arg_types
            parameter_values = []
            for parameter_type in parameter_types:
                parameter_values += [action_values[1 + self._parameter_registry[parameter_type]]]
            transformed_action = action.transform_action(observation, parameter_values)

        return [transformed_action]

    def convert_to_gym_action_spaces(self):
        vector = [0] * (1 + len(self._parameter_registry))
        vector[0] = self._num_actions
        for arg_type in self._parameter_registry:
            vector[1 + self._parameter_registry[arg_type]] = \
                retrieve_parameter_size_vector(arg_type,
                                               self._feature_screen_size,
                                               self._feature_minimap_size)
        return MultiDiscrete(vector)


class Action(ABC):
    """An abstract class for a default action which relies only on the policy network's output"""
    def __init__(self, arg_types, **kwargs):
        self._default_arguments = kwargs
        self._arg_types = arg_types
        self._feature_screen_size = kwargs['feature_screen_size'] if 'feature_screen_size' in kwargs else None
        self._feature_minimap_size = kwargs['feature_minimap_size'] if 'feature_minimap_size' in kwargs else None
        self._defaults = kwargs

    @abstractmethod
    def get_pysc2_action_ids(self):
        pass

    @abstractmethod
    def transform_action(self, observation, action_values):
        pass

    @property
    def arg_types(self):
        return self._arg_types


class AtomAction(Action):
    """A Class made to directly mirror pysc2 523 default actions"""
    def __init__(self, function_tuple, **kwargs):
        self.function_tuple = function_tuple
        arg_types = function_tuple.args
        super().__init__(arg_types, **kwargs)

    @classmethod
    def factory(cls, pysc2_function):
        """A Factory method to create an atom action based on pysc2 actions"""

        def __init__(self, **kwargs):
            super(self.__class__, self).__init__(pysc2_function, **kwargs)

        new_name = "".join(map(str.capitalize, pysc2_function.name.split('_'))) + "Action"
        return type(new_name, (AtomAction,), {'__init__': __init__})

    def get_pysc2_action_ids(self):
        return [self.function_tuple.id]

    def transform_action(self, observation, action_values):
        arg_values = []
        for arg_type in self._arg_types:
            if arg_type.name in self._defaults:
                arg_values += [self._defaults[arg_type.name]]
            else:
                arg_values += [action_values[self._arg_types[arg_type]]]
        return self.function_tuple(*arg_values)


class NoOpAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.no_op, **kwargs)


class SelectPointAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.select_point, **kwargs)


class SelectRectAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.select_point, **kwargs)


class SelectArmyAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.select_army, **kwargs)


class MoveScreenAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.Move_screen, **kwargs)


class MoveScreenAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.Move_screen, **kwargs)


class MoveScreenAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.Move_screen, **kwargs)
