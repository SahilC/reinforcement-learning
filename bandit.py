import numpy as np
class Bandit:
    def __init__(self, k):
        # k: number of bandit arms
        self.k = k
        
        # qstar: action values
        # self.qstar = []
        # for i in xrange(1, k+1):
        # 	v = np.random.normal(5 + i, 1)
        # 	self.qstar.append(v)
    
    def action(self, a):
        return np.random.normal(5 + a, 1)
