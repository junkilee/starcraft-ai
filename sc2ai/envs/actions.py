from abc import ABC, abstractmethod
from pysc2.lib import actions
from sc2ai.envs import game_info
from gym.spaces.multi_discrete import MultiDiscrete
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ActionVectorType(Enum):
    ACTION_TYPE = 1
    SCALAR = 2
    SPATIAL = 3


def retrieve_parameter_size_vector(arg_type, feature_screen_size, feature_minimap_size):
    if arg_type.sizes == (0, 0):
        if arg_type is actions.TYPES.screen or arg_type is actions.TYPES.screen2:
            return feature_screen_size ** 2
        elif arg_type is actions.TYPES.minimap:
            return feature_minimap_size ** 2
        else:
            raise NotImplementedError
    else:
        return np.prod(arg_type.sizes)


def translate_parameter_value(arg_type, value, feature_screen_size, feature_minimap_size):
    if arg_type.sizes == (0, 0):
        if arg_type is actions.TYPES.screen or arg_type is actions.TYPES.screen2:
            return [value // feature_screen_size, value % feature_screen_size]
        elif arg_type is actions.TYPES.minimap:
            return [value // feature_minimap_size, value % feature_minimap_size]
        else:
            raise NotImplementedError
    else:
        return [value]


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
        self._current_available_actions = [False] * self._num_actions

    @abstractmethod
    def convert_to_gym_action_spaces(self):
        pass

    @abstractmethod
    def transform_action(self, observation, action_values):
        pass

    @abstractmethod
    def get_action_spec_and_action_mask(self):
        pass

    def is_action_available(self, action_index):
        if action_index < 0 or action_index >= self._num_actions:
            raise Exception("action index is out of range.")
        return self._current_available_actions[action_index]

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
            self._current_available_actions[i] = not check


class DefaultActionSet(ActionSet):
    """Store a list of default PySC2 actions in this set.
    Uses a shared parameter space called _parameter_registry."""

    def __init__(self, action_list, reorder_action_id=False,
                 feature_screen_size=game_info.feature_screen_size,
                 feature_minimap_size=game_info.feature_minimap_size):
        super().__init__(action_list)
        self._reorder_action_id = reorder_action_id
        self._current_num_actions = self._num_actions
        self._parameter_registry, self._action_mask = self.register_argument_types()
        self._feature_screen_size = feature_screen_size
        self._feature_minimap_size = feature_minimap_size
        self._no_op_action = NoOpAction()

    def register_argument_types(self):
        registry = {}
        count = 0
        for action in self._action_list:
            for parameter in action.arg_types:
                if not (parameter.name in action.defaults) and not (parameter in registry):
                    registry[parameter] = count
                    count += 1
        action_mask = np.zeros((len(self._action_list), count + 1))
        for i, action in enumerate(self._action_list):
            action_mask[i][0] = 1
            for parameter in action.arg_types:
                if not (parameter.name in action.defaults):
                    action_mask[i][registry[parameter] + 1] = 1
        return registry, action_mask

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
        # print("action value : ", action_values)
        action_id = action_values[0]

        if self._reorder_action_id:
            raise NotImplementedError()
        else:
            if self.is_action_available(action_id):
                if action_id < 0 or action_id > len(self._action_list):
                    logger.error("The wrong action ID %d from the network output", action_id)
                    raise Exception("The wrong action ID. {}".format(action_id))
                action = self._action_list[action_id]
                parameter_types = action.arg_types
                parameter_values = []
                for parameter_type in parameter_types:
                    if parameter_type.name in action.defaults:
                        parameter_values += [action.defaults[parameter_type.name]]
                    else:
                        parameter_values += [action_values[1 + self._parameter_registry[parameter_type]]]
                transformed_action = action.transform_action(observation, parameter_values)
            else:
                transformed_action = self._no_op_action.transform_action(observation, [])

        return [transformed_action]

    def convert_to_gym_action_spaces(self):
        vector = [0] * (1 + len(self._parameter_registry))
        vector[0] = self._num_actions
        for arg_type in self._parameter_registry:
            vector[1 + self._parameter_registry[arg_type]] = \
                retrieve_parameter_size_vector(arg_type,
                                               self._feature_screen_size,
                                               self._feature_minimap_size)
        # print(vector)
        return MultiDiscrete(vector)

    def get_action_spec_and_action_mask(self):
        action_spec = list()
        action_spec.append((ActionVectorType.ACTION_TYPE, self._num_actions))
        for arg_type in self._parameter_registry:
            size = retrieve_parameter_size_vector(arg_type,
                                                  self._feature_screen_size,
                                                  self._feature_minimap_size)
            _type = None
            if arg_type is actions.TYPES.screen or arg_type is actions.TYPES.screen2 or \
                    arg_type is actions.TYPES.minimap:
                _type = ActionVectorType.SPATIAL
            else:
                _type = ActionVectorType.SCALAR
            action_spec.append((_type, size))
        return action_spec, self._action_mask

    def report(self):
        print("---- Action Parameters List ----")
        for arg_type in self._parameter_registry:
            print(arg_type.name, self._parameter_registry[arg_type])
        print("--------------------------------")


class Action(ABC):
    """An abstract class for a default action which relies only on the policy network's output"""

    def __init__(self, arg_types, **kwargs):
        self._arg_types = arg_types
        self._feature_screen_size = \
            kwargs['feature_screen_size'] if 'feature_screen_size' in kwargs else game_info.feature_screen_size
        self._feature_minimap_size = \
            kwargs['feature_minimap_size'] if 'feature_minimap_size' in kwargs else game_info.feature_minimap_size
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

    @property
    def defaults(self):
        return self._defaults


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
        # print(action_values)
        # print(self.__class__)
        # print(self._arg_types)
        for i, arg_type in enumerate(self._arg_types):
            if arg_type.name in self._defaults:
                arg_values += [self._defaults[arg_type.name]]
            else:
                arg_values += [translate_parameter_value(arg_type, action_values[i], self._feature_screen_size,
                                                         self._feature_minimap_size)]
        # print(arg_values)
        return self.function_tuple(*arg_values)


class NoOpAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.no_op, **kwargs)


class SelectPointAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.select_point, **kwargs)


class SelectRectAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.select_rect, **kwargs)


class SelectArmyAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.select_army, **kwargs)


class MoveScreenAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.Move_screen, **kwargs)


class MoveScreenAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.Move_screen, **kwargs)


class AttackScreenAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.Attack_screen, **kwargs)


class TrainSCVAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.Train_SCV_quick, **kwargs)


class TrainMarineAction(AtomAction):
    def __init__(self, **kwargs):
        super().__init__(actions.FUNCTIONS.Train_Marine_quick, **kwargs)
