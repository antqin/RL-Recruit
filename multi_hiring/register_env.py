import gymnasium as gym
from multi_hiring.multi_hiring import MultiHiringEnv

gym.register(
    id='MultiHiring-v0',
    entry_point='multi_hiring.multi_hiring:MultiHiringEnv',
)


