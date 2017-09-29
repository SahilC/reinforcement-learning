import numpy as np
class Bandit:
    def __init__(self, k):
        # k: number of bandit arms
        self.k = k
        
        # qstar: action values
        self.qstar = np.random.normal(size=k)
    
    def action(self, a):
        return np.random.normal(loc=self.qstar[a])