from gymnasium.spaces import Discrete, Tuple, Box
import env_params
import numpy as np

test = Box(low=-np.inf, high=np.inf, shape=(3,))

print(test.sample())