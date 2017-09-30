import numpy as np
import matplotlib.pyplot as plt

from greedy_bandit import *

if __name__ == '__main__':
	# 10 arms 
	k = 10
	numsteps = 1000
	numtasks = 200
	epislon = [0, 0.01, 0.1]
	avgR = np.zeros((numsteps, ))
	for e in epislon:
		for task in range(2000):
		    bandit_task = greedy_action_selection(k, numsteps, e)
		    avgR += bandit_task['R']
	avgR /= numtasks

	plt.plot(avgR)
	plt.ylabel('Average reward')
	plt.xlabel('Steps')
	plt.xlim(-5)
	plt.show()
