import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import multi_hiring.register_env
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

# Parallel environments
eval_env = VecNormalize(make_vec_env('MultiHiring-v0', n_envs=4, env_kwargs={"render_mode": None}), norm_obs_keys=["avg_interview_score"])

train_env = VecNormalize(make_vec_env('MultiHiringTrain-v0', n_envs=4, env_kwargs={"render_mode": None}), norm_obs_keys=["avg_interview_score"])

model = DQN("MultiInputPolicy", train_env, verbose=1, gamma=1, tensorboard_log="./tensorboard_dqn")
model.learn(total_timesteps=500_000, log_interval=10_000)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100_000, deterministic=False)
print("Mean reward:", mean_reward, "Std reward:", std_reward)