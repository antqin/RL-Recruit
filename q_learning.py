import numpy as np
import random
import model

# Constants
num_candidates = 1
initial_budget = 500000
interview_cost = 10000
hiring_cost = 100000
value_range = (50000, 500000)

# Discretization parameters
budget_levels = 10  # Example discretization
interview_levels = 10  # Maximum number of interviews per candidate

# Create candidates
candidates = {i: model.Candidate(random.randint(*value_range)) for i in range(num_candidates)}
for idx, candidate in candidates.items():
    print("Candidate " + str(idx) + " true value:")
    print(candidate.true_value)

# Initialize the environment
env = model.HiringEnvironment(initial_budget, candidates, interview_cost, hiring_cost)

# Number of states and actions
num_states = budget_levels * interview_levels
num_actions = 2  # INTERVIEW, HIRE

# Initialize Q-table
q_table = np.zeros((num_states, num_actions))

def encode_state(state):
    # Example: Encode state based on budget level and total interviews conducted
    budget_index = int(state.budget / initial_budget * (budget_levels - 1))
    interviews_conducted = sum(c.interview_count for c in state.candidates_info.values())
    interviews_index = min(interviews_conducted, interview_levels - 1)

    return budget_index * interview_levels + interviews_index

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.1
num_episodes = 50000
episode_rewards = []

def simulate_reward(state, action_type):
    temp_state = state.copy()
    action = model.Action(action_type, random.choice(list(temp_state.candidates_info.keys())))
    temp_next_state, simulated_reward, _ = env.step(action)
    return simulated_reward

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_index = encode_state(state)

        interview_reward = simulate_reward(state, model.Action.INTERVIEW)
        hire_reward = simulate_reward(state, model.Action.HIRE)
        
        if interview_reward > hire_reward:
            action_type = model.Action.INTERVIEW
        else:
            action_type = model.Action.HIRE
        
        candidate_id = random.choice(list(state.candidates_info.keys()))

        action = model.Action(action_type, candidate_id)

        # Take action and observe next state and reward
        next_state, reward, done = env.step(action)
        next_state_index = encode_state(next_state)

        # Q-learning update
        action_index = 0 if action_type == model.Action.INTERVIEW else 1
        best_next_action = np.argmax(q_table[next_state_index])
        td_target = reward + discount_factor * q_table[next_state_index][best_next_action]
        td_error = td_target - q_table[state_index][action_index]
        q_table[state_index][action_index] += learning_rate * td_error

        state = next_state
        # print(candidates[candidate_id].estimated_value)
        total_reward += reward

    episode_rewards.append(total_reward)
    # print(f"Episode {episode + 1}: Total Reward = {total_reward}")

def evaluate_policy(env, policy, num_trials=100):
    total_rewards = []

    for i in range(num_trials):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_index = encode_state(state)
            action_index = policy[state_index]
            action_type = model.Action.INTERVIEW if action_index == 0 else model.Action.HIRE

            candidate_id = random.choice(list(state.candidates_info.keys()))  # Choosing a candidate

            action = model.Action(action_type, candidate_id)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

    return np.mean(total_rewards)

## After all episodes
print("\nBest Policy (Action for each State):")
policy = np.argmax(q_table, axis=1)
print(q_table)
print(policy)
average_reward = evaluate_policy(env, policy)
print(f"Average Reward achieved by the policy: {average_reward}")

def decode_state(index, budget_levels, interview_levels):
    budget_index = index // interview_levels
    interviews_index = index % interview_levels
    return budget_index, interviews_index

# Print policy for each state
for state_index in range(num_states):
    action = "INTERVIEW" if policy[state_index] == 0 else "HIRE"

    budget_index, interviews_index = decode_state(state_index, budget_levels, interview_levels)

    # Calculate the corresponding budget and interview count
    budget = (budget_index / (budget_levels - 1)) * initial_budget
    interviews_count = interviews_index

    print(f"State {state_index} (Budget: {budget}, Interviews: {interviews_count}): Best Action = {action}")
