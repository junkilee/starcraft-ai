import pytest
from .spaces import ListActionSpace
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
import logging

class TestListActionSpace:
    def test_first(self):
        list_action_space = ListActionSpace([("default", None), ("first", Tuple((Discrete(2), Box(low = 0, high = 255, shape=(1, 1)))))])
        print(list_action_space.sample())
        for i in range(10):
            sample = list_action_space.sample()
            jsonified = list_action_space.to_jsonable([sample])
            print(sample, jsonified)
            print(sample, list_action_space.from_jsonable(jsonified))
        assert False
