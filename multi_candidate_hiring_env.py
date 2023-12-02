import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete, Tuple
import env_params


class MultiCandidateHiringEnv(gym.Env):

    def __init__(self, render_mode=None):
        # We have 3 actions: "hire", "reject", and "interview".
        self.action_space = spaces.Discrete(3)

        # The state is represented by a 3-tuple:
        # (avg interview score, num_interviews, candidate_number)
        self.observation_space = Tuple((Box(low=-np.inf, high=np.inf, shape=(1,)), Discrete(50, start=0), Discrete(env_params.NUM_CANDIDATES)))