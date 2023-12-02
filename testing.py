from gymnasium.spaces import Discrete, Tuple, Box, Dict
import env_params
import numpy as np

test = Dict({"avg_interview_score": Box(low=-np.inf, high=np.inf, shape=(1,)), "interview_count": Discrete(100), "candidate_number": Discrete(env_params.NUM_CANDIDATES)})

print(test.sample())