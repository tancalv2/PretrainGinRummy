''' Register new environments
'''
from rlcard.envs.env import Env
from rlcard.envs.vec_env import VecEnv
from rlcard.envs.registration import register, make

register(
    env_id='gin-rummy',
    entry_point='rlcard.envs.gin_rummy:GinRummyEnv',
)
