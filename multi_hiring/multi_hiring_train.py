import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box, Discrete, Tuple
import env_params
import collections
from multi_hiring.multi_hiring import MultiHiringEnv


class MultiHiringTrainEnv(MultiHiringEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
    
    @property
    def observation(self):
        normalized_avg_interview_score = (self.state[0] - env_params.AVG_CANDIDATE_VALUE) / env_params.CANDIDATE_VALUE_VARIANCE
        return collections.OrderedDict([
            ("avg_interview_score", np.array([normalized_avg_interview_score], dtype=np.float32)),
            ("interview_count", self.state[1]),
            ("candidate_number", self.state[2])
        ])

    def step(self, action):
        # inheret from MultiVCEnvironment
        observation, reward, terminated, truncated, _ = super().step(action)
        if action == -1:
            num_times_interviewed = observation["interview_count"]
            reward += 1000 / (num_times_interviewed + 1)


        return observation, reward, terminated, truncated, _
    
    def reset(self, seed=None, options=None):
        # inheret from MultiVCEnvironment
        return super().reset(seed)
    
    def render(self):
        pass