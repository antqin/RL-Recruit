import numpy as np
from collections import defaultdict

def discretize(value, increment=500):
    return round(value / increment) * increment


class CandidateDistribution:
    def __init__(self, avg_candidate_value, value_variance):
        self.avg_candidate_value = avg_candidate_value
        self.value_variance = value_variance

    def sample(self, discretized=True):
        value = np.random.normal(self.avg_candidate_value, self.value_variance)
        return discretize(value) if discretized else value
    

# Environment parameters
INTERVIEW_COST = 300 # cost of interviewing a candidate
SALARY = 150000 # salary of the candidate (we must pay this amount upon hire)
AVG_CANDIDATE_VALUE = 130000 # average value of a candidate
CANDIDATE_VALUE_VARIANCE = 30000 # variance of candidate value
INTERVIEW_VARIANCE = 10000 # variance of interview score
CANDIDATE_DISTRIBUTION = CandidateDistribution(AVG_CANDIDATE_VALUE, CANDIDATE_VALUE_VARIANCE) # the candidate we are interviewing will be samples from this distribution


class HiringEnvironment:
    def __init__(self, interview_cost, salary, candidate_distribution, interview_variance):
        self.interview_cost = interview_cost
        self.salary = salary
        self.interview_variance = interview_variance
        self.true_candidate_value = candidate_distribution.sample()

        self.state = [AVG_CANDIDATE_VALUE, 0]  # (avg interview score, num_interviews)
        self.actions = ["interview", "hire", "reject"]
        self.reward = 0
        self.terminated = False

    def step(self, action):
        if action == "interview":
            interview_score = np.random.normal(self.true_candidate_value, self.interview_variance)
            self.state = [discretize((self.state[0] * self.state[1] + interview_score) / (self.state[1] + 1)), self.state[1] + 1]
            self.reward = -self.interview_cost
        elif action == "hire":
            self.reward = self.true_candidate_value - self.salary
            self.terminated = True
        else:
            self.terminated = True

        return tuple(self.state), self.reward, self.terminated

    def reset(self):
        self.state = [AVG_CANDIDATE_VALUE, 0]
        self.reward = 0
        self.terminated = False
        return tuple(self.state)
    
# Creating the environment
env = HiringEnvironment(INTERVIEW_COST, SALARY, CANDIDATE_DISTRIBUTION, INTERVIEW_VARIANCE)

# Example of environment interaction
state = env.reset()
next_state, reward, done = env.step("interview")
print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")

# This setup is ready for Q-learning implementation, where the agent will choose actions and learn from interactions.
import random

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.4):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: [0, 0, 0])  # (avg interview score, num_interviews) -> [Q-value of interview, Q-value of hire, Q-value of reject]

    def choose_action(self, state):
        # Implementing epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)  # Explore
        else:
            return self.env.actions[np.argmax(self.q_table[state])]  # Exploit

    def learn(self, state, action, reward, next_state):
        # Implementing the Q-learning update rule
        action_index = self.env.actions.index(action)
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state, action_index]
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state, action_index] = new_q

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state

# Set the number of episodes for training
num_episodes = 4

# Create the Q-learning agent with the environment
ql_agent = QLearningAgent(env)

# Train the agent
ql_agent.train(num_episodes)

# Display the final Q-table
print(ql_agent.q_table)

# Extracting and printing the optimal policy from the Q-table
optimal_policy = [env.actions[np.argmax(action_values)] for action_values in ql_agent.q_table.values()]

print(optimal_policy)
