import numpy as np
from greedy_bandit import *

# import matplotlib.pyplot as plt

if __name__ == '__main__':
	# 10 arms 
	k = 10
	numsteps = 100000
	numtasks = 200
	epislon = [0, 0.01, 0.1]
	avgR = np.zeros((numsteps, ))
	
	for e in epislon:
		print 'Starting'

		for task in range(numtasks):
		    bandit_task = greedy_action_selection(k, numsteps, e)
		    avgR += bandit_task['R']
		avgR /= numtasks

		file_name = 'results_'+str(e)+'.npy'
		np.save(file_name, avgR)
		
	# plt.plot(avgR)
	# plt.ylabel('Average reward')
	# plt.xlabel('Steps')
	# plt.xlim(-5)
	# plt.show()
