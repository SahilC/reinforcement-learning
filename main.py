import numpy as np
import matplotlib.pyplot as plt

from greedy_bandit import *

if __name__ == '__main__':
	k = 10
	numsteps = 1000
	numtasks = 2000

	avgR = np.zeros((numsteps, ))
	for task in range(2000):
	    bandit_task = greedy_action_selection(k,numsteps)
	    avgR += bandit_task['R']
	avgR /= numtasks

	plt.plot(avgR)
	plt.ylabel('Average reward')
	plt.xlabel('Steps')
	plt.xlim(-5)
	plt.show()