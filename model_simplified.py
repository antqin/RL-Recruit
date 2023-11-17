import numpy as np

# TODO:
# 1) change the state to be a tuple of (avg interview score, num_interviews)
# 2) estimated value should come from the same distribution every time - uncertaintiy is expressed in num_interviews, not explicitly in different variance
# 3) discritization of candidate value so that we have more managable state space
# 4) add rejecting the canddiate to the action space
# 5) parameters should reflect real world

class HiringEnvironment:
    def __init__(self, interview_cost, hiring_cost, candidate_true_value, starting_budget, value_variance):
        self.interview_cost = interview_cost
        self.hiring_cost = hiring_cost
        self.candidate_true_value = candidate_true_value
        self.starting_budget = starting_budget
        self.value_variance = value_variance  # Variance in value estimate decreases with more interviews
        self.state = 0  # Starting with 0 interviews
        self.terminal_state = 11  # Represents the "HIRE" state
        self.total_states = self.terminal_state + 1  # 0 to 10 interviews + HIRE state
        self.actions = ["Interview", "Hire"]
        self.estimated_value = 0
        self.reward = 0
        self.done = False

    def step(self, action):
        if action == "Interview" and self.state < 10:
            self.state += 1
            self.estimated_value = np.random.normal(self.candidate_true_value, self.value_variance / self.state)
            self.reward = -self.interview_cost
            self.done = False
        elif action == "Hire":
            self.state = self.terminal_state
            self.reward = self.estimated_value - self.hiring_cost  # Reward based on estimated candidate value
            self.done = True
        else:
            self.reward = 0
            self.done = True

        return self.state, self.reward, self.done

    def reset(self):
        self.state = 0
        self.estimated_value = 0
        self.reward = 0
        self.done = False
        return self.state

# Example parameters
interview_cost = 100
hiring_cost = 5000
candidate_true_value = 10000
starting_budget = 10000
value_variance = 5000

# Creating the environment
env = HiringEnvironment(interview_cost, hiring_cost, candidate_true_value, starting_budget, value_variance)

# Example of environment interaction
state = env.reset()
next_state, reward, done = env.step("Interview")
print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")

# This setup is ready for Q-learning implementation, where the agent will choose actions and learn from interactions.
import random

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.4):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((env.total_states, len(env.actions)))

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
num_episodes = 50000

# Create the Q-learning agent with the environment
ql_agent = QLearningAgent(env)

# Train the agent
ql_agent.train(num_episodes)

# Display the final Q-table
print(ql_agent.q_table)

# Extracting and printing the optimal policy from the Q-table
optimal_policy = [env.actions[np.argmax(action_values)] for action_values in ql_agent.q_table]

print(optimal_policy)
