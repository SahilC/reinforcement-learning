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

    def loss(self, delta, t, numsteps):
    	loss = [np.random.binomial(1, 0.5, 1) for i in xrange(8)]
    	
    	loss.append(np.random.binomial(1, 0.5 - delta, 1))
    	if t < (numsteps/2):
    		loss.append(np.random.binomial(1, 0.5 + delta, 1))
    	else:
    		loss.append(np.random.binomial(1, 0.5 - 2*delta, 1))
    	loss = np.array(loss)
    	return loss

