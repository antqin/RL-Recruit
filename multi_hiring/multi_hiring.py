import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box, Discrete
import env_params
import collections


class MultiHiringEnv(gym.Env):

    def __init__(self, render_mode=None):
        # We have 3 actions: interview, reject, hire
        self.action_space = spaces.Discrete(3)

        # The state is represented by a 3-tuple:
        # (avg interview score, num_interviews, candidate_number)
        self.observation_space = Dict({"avg_interview_score": Box(low=-np.inf, high=np.inf, shape=(1,)), "interview_count": Discrete(100), "candidate_number": Discrete(env_params.NUM_CANDIDATES)})

        self.true_candidate_value = env_params.CANDIDATE_DISTRIBUTION.sample()
        self.salary = env_params.SALARY
        self.interview_cost = env_params.INTERVIEW_COST
        self.interview_variance = env_params.INTERVIEW_VARIANCE
        self.state = [env_params.AVG_CANDIDATE_VALUE, 0, 0]

    @property
    def observation(self):
        normalized_avg_interview_score = (self.state[0] - env_params.AVG_CANDIDATE_VALUE) / env_params.CANDIDATE_VALUE_VARIANCE
        return collections.OrderedDict([
            ("avg_interview_score", np.array([normalized_avg_interview_score], dtype=np.float32)),
            ("interview_count", self.state[1]),
            ("candidate_number", self.state[2])
        ])

    def step(self, action):
        terminated = False
        reward = 0

        if action == 0: # interview
            interview_score = np.random.normal(self.true_candidate_value, self.interview_variance)
            self.state = [(self.state[0] * self.state[1] + interview_score) / (self.state[1] + 1), self.state[1] + 1, self.state[2]]
            if self.state[1] >= 99:
                terminated = True
        elif action == 1: # hire
            reward = self.true_candidate_value - self.salary - self.state[1] * self.interview_cost
            terminated = True
        else: # reject
            if self.state[2] == env_params.NUM_CANDIDATES - 1: # if we have interviewed all candidates
                terminated = True
            else:
                self.true_candidate_value = env_params.CANDIDATE_DISTRIBUTION.sample()
                self.state = [env_params.AVG_CANDIDATE_VALUE, 0, self.state[2] + 1]
            reward = - self.state[1] * self.interview_cost

        return self.observation, reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        self.true_candidate_value = env_params.CANDIDATE_DISTRIBUTION.sample()
        self.state = [np.random.normal(self.true_candidate_value, self.interview_variance), 1, 0]
        return self.observation, {}
    
    def render(self):
        pass