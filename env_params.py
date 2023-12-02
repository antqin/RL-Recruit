import numpy as np

def discretize(value, increment=1000):
    return round(value / increment) * increment

class CandidateDistribution:
    def __init__(self, avg_candidate_value, value_variance):
        self.avg_candidate_value = avg_candidate_value
        self.value_variance = value_variance

    def sample(self, discretized=True):
        value = np.random.normal(self.avg_candidate_value, self.value_variance)
        return discretize(value) if discretized else value

INTERVIEW_COST = 500 # cost of interviewing a candidate
SALARY = 150000 # salary of the candidate (we must pay this amount upon hire)
AVG_CANDIDATE_VALUE = 140000 # average value of a candidate
CANDIDATE_VALUE_VARIANCE = 30000 # variance of candidate value
INTERVIEW_VARIANCE = 30000 # variance of interview score
CANDIDATE_DISTRIBUTION = CandidateDistribution(AVG_CANDIDATE_VALUE, CANDIDATE_VALUE_VARIANCE) # the candidate we are interviewing will be samples from this distribution
NUM_CANDIDATES = 3 # number of candidates we can interview