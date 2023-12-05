import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import multi_hiring.register_env
import numpy as np

# Parallel environments
eval_env = make_vec_env('MultiHiring-v0', n_envs=4, env_kwargs={"render_mode": None})

train_env = make_vec_env('MultiHiringTrain-v0', n_envs=4, env_kwargs={"render_mode": None})

model = PPO("MultiInputPolicy", train_env, verbose=1, gamma=1)
model.learn(total_timesteps=500_000)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100_000, deterministic=False)
print("Mean reward:", mean_reward, "Std reward:", std_reward)