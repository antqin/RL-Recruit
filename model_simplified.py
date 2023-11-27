import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def discretize(value, increment=1000):
    return round(value / increment) * increment


class CandidateDistribution:
    def __init__(self, avg_candidate_value, value_variance):
        self.avg_candidate_value = avg_candidate_value
        self.value_variance = value_variance

    def sample(self, discretized=True):
        value = np.random.normal(self.avg_candidate_value, self.value_variance)
        return discretize(value) if discretized else value
    

# Environment parameters
INTERVIEW_COST = 500 # cost of interviewing a candidate
SALARY = 150000 # salary of the candidate (we must pay this amount upon hire)
AVG_CANDIDATE_VALUE = 140000 # average value of a candidate
CANDIDATE_VALUE_VARIANCE = 30000 # variance of candidate value
INTERVIEW_VARIANCE = 40000 # variance of interview score
CANDIDATE_DISTRIBUTION = CandidateDistribution(AVG_CANDIDATE_VALUE, CANDIDATE_VALUE_VARIANCE) # the candidate we are interviewing will be samples from this distribution
NUM_CANDIDATES = 5 # number of candidates we can interview

class HiringEnvironment:
    def __init__(self, interview_cost, salary, interview_variance):
        self.interview_cost = interview_cost
        self.salary = salary
        self.interview_variance = interview_variance
        self.true_candidate_value = CANDIDATE_DISTRIBUTION.sample()

        self.state = [AVG_CANDIDATE_VALUE, 0, 0]  # (avg interview score, num_interviews, candidate_number)
        self.actions = ["hire", "interview", "reject"]
        self.reward = 0
        self.terminated = False

    def step(self, action):
        if action == "interview":
            interview_score = np.random.normal(self.true_candidate_value, self.interview_variance)
            self.state = [discretize((self.state[0] * self.state[1] + interview_score) / (self.state[1] + 1)), self.state[1] + 1, self.state[2]]
            self.reward -= self.interview_cost
        elif action == "hire":
            self.reward += self.true_candidate_value - self.salary
            self.terminated = True  
        else:
            if self.state[2] == NUM_CANDIDATES - 1:
                self.terminated = True
            else:
                self.true_candidate_value = CANDIDATE_DISTRIBUTION.sample()
                self.state = [AVG_CANDIDATE_VALUE, 0, self.state[2] + 1]

        return tuple(self.state), self.reward, self.terminated

    def reset(self):
        self.state = [AVG_CANDIDATE_VALUE, 0, 0] # prior belief about the candidate
        self.reward = 0
        self.terminated = False
        self.true_candidate_value = CANDIDATE_DISTRIBUTION.sample()
        return tuple(self.state)
    
# Creating the environment
env = HiringEnvironment(INTERVIEW_COST, SALARY, INTERVIEW_VARIANCE)

# Example of environment interaction
state = env.reset()
next_state, reward, done = env.step("interview")
print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")

# This setup is ready for Q-learning implementation, where the agent will choose actions and learn from interactions.
import random

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=1, epsilon=0.3):
        self.env = env
        self.lr = learning_rate
        self.inv_lr = 1 - learning_rate
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
        current_q = self.q_table[state][action_index]
        new_q = self.inv_lr * current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action_index] = new_q
    
    def train(self, num_episodes, evaluation_interval=50000, num_evaluation_episodes=10000):
        evaluation_results = []
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state

            # Evaluate every 'evaluation_interval' episodes
            if episode % evaluation_interval == 0:
                avg_reward = self.evaluate_policy(num_evaluation_episodes)
                evaluation_results.append((episode, avg_reward))
                print(f"Episode: {episode}, Average Reward: {avg_reward}")

        return evaluation_results

    def evaluate_policy(self, num_episodes):
        total_reward = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                # Choose the best action from Q-table
                action = self.env.actions[np.argmax(self.q_table[state])]
                # Always interview if candidate hasn't been interviewed yet
                if state[1] == 0:
                    action = "interview"
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                state = next_state
        return total_reward / num_episodes
    

# Greedy agent that will do one interview and then hire if the interview score is above the salary
class GreedyAgent:
    def __init__(self, env):
        self.env = env

    def choose_action(self, state):
        if state[1] == 0:
            return "interview"
        elif state[0] > SALARY:
            return "hire"
        else:
            return "reject"

    def evaluate_policy(self, num_episodes):
        total_reward = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                state = next_state
        return total_reward / num_episodes

# Set the number of episodes for training
num_episodes = 1000000

# Create the Q-learning agent with the environment
ql_agent = QLearningAgent(env)

# Train the agent and capture evaluation results
evaluation_results = ql_agent.train(num_episodes, evaluation_interval=50000, num_evaluation_episodes=10000)

# Plotting the results
episodes, avg_rewards = zip(*evaluation_results)
plt.plot(episodes, avg_rewards, marker='o')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs. Training Episodes')

# Save the plot to a file
plt.savefig('q-learning-training.png')

# Evaluate the greedy agent
greedy_agent = GreedyAgent(env)
avg_reward = greedy_agent.evaluate_policy(10000)
print(f"Average reward of greedy agent: {avg_reward}")
plt.show()