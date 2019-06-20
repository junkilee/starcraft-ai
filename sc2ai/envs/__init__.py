import gym
from gym.envs.registration import register
from sc2ai.envs import minigames


register(
    id='SingleAgentSC2Env-v0',
    entry_point='sc2ai.envs.sc2env:SingleAgentSC2Env',
    kwargs={}
)


MAP_ENV_MAPPINGS = {
    "DefeatZerglings": minigames.DefeatZerglingsEnv,
    "CollectMineralAndGas": minigames.CollectMineralAndGasEnv,
    "MoveToBeacon": minigames.MoveToBeaconEnv,
    "BuildMarines": minigames.BuildMarinesEnv
}


def make_sc2env(**kwargs):
    """

    Args:
        **kwargs:

    Returns:

    """
    if kwargs["map"] not in MAP_ENV_MAPPINGS:
        raise Exception("The map is unknown and not registered.")
    else:
        cls = MAP_ENV_MAPPINGS[kwargs["map"]]

    return cls(**kwargs)