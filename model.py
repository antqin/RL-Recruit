import random
class State:
    def __init__(self, budget, candidates_info, time=None):
        self.budget = budget
        self.candidates_info = candidates_info  # Dictionary or list of candidate objects
        self.time = time  # Optional, for dynamic candidate pool

    def copy(self):
        # Manually copy the candidates
        new_candidates_info = {cid: Candidate(candidate.true_value, candidate.interview_count) 
                               for cid, candidate in self.candidates_info.items()}
        return State(self.budget, new_candidates_info, self.time)
        
class Action:
    INTERVIEW = 0
    HIRE = 1

    def __init__(self, action_type, candidate_id=None):
        self.action_type = action_type
        self.candidate_id = candidate_id  # Relevant for INTERVIEW and HIRE actions

class Candidate:
    def __init__(self, true_value, interview_count=0):
        self.true_value = true_value  # Candidate's true value if hired
        self.interview_count = interview_count
        self.estimated_value = 0  # Initial estimate

    def update_estimated_value(self):
        noise = random.gauss(0, 1 / (self.interview_count + 1))  # Decreasing noise with more interviews
        self.estimated_value = self.true_value + noise
        return self.estimated_value

class HiringEnvironment:
    def __init__(self, initial_budget, candidates, interview_cost, hiring_cost):
        self.initial_budget = initial_budget
        self.state = State(initial_budget, candidates)
        self.interview_cost = interview_cost # Cost per interview
        self.hiring_cost = hiring_cost  # Promised salary of hire
        self.hire_made = False

    def transition_function(self, state, action):
        next_state = state.copy()
        if action.action_type == Action.INTERVIEW:
            candidate = next_state.candidates_info[action.candidate_id]
            candidate.interview_count += 1
            candidate.update_estimated_value()
            next_state.budget -= self.interview_cost
        elif action.action_type == Action.HIRE:
            next_state.budget -= self.hiring_cost
        return next_state

    def reward_function(self, state, action, next_state):
        if action.action_type == Action.INTERVIEW:
            candidate = next_state.candidates_info[action.candidate_id]
            interview_count = candidate.interview_count
            estimated_value = candidate.estimated_value

            # Information gain factor decreases with more interviews
            information_gain = 1 / (interview_count + 1)  # Diminishing returns with more interviews

            # Scaling factor based on estimated value (capped to avoid excessive rewards)
            scaling_factor = min(estimated_value * 0.01, self.interview_cost)

            # Reward for interviewing combines information gain and estimated value, minus cost
            reward = next_state.budget
            print("Interview reward: " + str(reward))

        elif action.action_type == Action.HIRE:
            candidate = next_state.candidates_info[action.candidate_id]
            estimated_value = candidate.update_estimated_value()
            # print(candidate.update_estimated_value())
            # print("estimated value = " + str(estimated_value))
            # print("num interviews = " + str(candidate.interview_count))
            reward = next_state.budget + estimated_value
            self.hire_made = True
            print("Hire reward: " + str(reward))

        return reward



    def step(self, action):
        # Apply transition_function to update the state based on the action
        next_state = self.transition_function(self.state, action)

        # Update hire_made flag
        self.hire_made = action.action_type == Action.HIRE

        # Calculate the reward
        reward = self.reward_function(self.state, action, next_state)

        # Determine if the episode is done (e.g., a hiring decision is made)
        done = self.check_if_done(next_state)

        # Update the current state
        self.state = next_state
        return next_state, reward, done

    def check_if_done(self, state):
        # Check if a hire has been made
        if self.hire_made:
            return True

        # Check if the budget is too low to continue
        budget_too_low = state.budget < self.interview_cost or state.budget < self.hiring_cost
        if budget_too_low:
            return True

        return False

    def reset(self):
        # Reset the budget to the initial value
        self.state.budget = self.initial_budget

        # Reset each candidate's state
        for candidate in self.state.candidates_info.values():
            candidate.interview_count = 0
            candidate.estimated_value = 0
            candidate.update_estimated_value()  # Update the estimated value based on the reset interview count

        # Reset the hire_made flag
        self.hire_made = False

        # Return the reset state
        return self.state
