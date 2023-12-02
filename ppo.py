import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import multi_hiring.register_env
import numpy as np

# Parallel environments
vec_env = make_vec_env('MultiHiring-v0', n_envs=4, env_kwargs={"render_mode": None})

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("ppo_multihiring")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
print("Mean reward:", mean_reward, "Std reward:", std_reward)


# print out action given 100 random states
for i in range(100):
    # generate random state
    interview_count = np.random.randint(0, 10)
    avg_interview_score = np.random.uniform(50000, 300000)
    candidate_number = np.random.randint(0, 2)

    state = np.array([avg_interview_score, interview_count, candidate_number])
    print("State:", state)
    action, _ = model.predict(state)
    print("Action:", action)