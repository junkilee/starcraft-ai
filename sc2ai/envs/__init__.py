from gym.envs.registration import register

register(
    id='SC2Env-v0',
    entry_point='sc2ai.envs:SC2Env',
    kwargs={}
)
