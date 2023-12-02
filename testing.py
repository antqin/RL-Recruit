from gymnasium.spaces import Discrete, Tuple, Box
import env_params
import numpy as np

test = Tuple((Box(low=-np.inf, high=np.inf, shape=(1,)), Discrete(50, start=0), Discrete(env_params.NUM_CANDIDATES)))

print(test.sample())