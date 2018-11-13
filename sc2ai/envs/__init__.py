from gym.envs.registration import register

register(
    id='SC2Game-v0',
    entry_point='sc2gym.envs:SC2GameEnv',
    kwargs={}
)
