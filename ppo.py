import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import multi_hiring.register_env
import numpy as np

# Parallel environments
vec_env = make_vec_env('MultiHiring-v0', n_envs=4, env_kwargs={"render_mode": None})

model = PPO("MultiInputPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=500_000)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
print("Mean reward:", mean_reward, "Std reward:", std_reward)