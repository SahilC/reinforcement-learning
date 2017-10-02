import numpy as np
from greedy_bandit import *
from ucb import *
from visualize import *

def run(k, numsteps, epislon):
	
	
	numtasks = 200
	avgR = np.zeros((numsteps, ))

	avgA = np.zeros((k, ))
	
	for e in epislon:
		print 'Starting'

		for task in range(numtasks):
		    # bandit_task = greedy_action_selection(k, numsteps, e)
		    bandit_task = ucb_action_selection(k, numsteps, e)
		    avgR += bandit_task['R']

		    for t in range(numsteps):
		    	a = bandit_task['A'][t]
		    	avgA[int(a)] += 1
		    
		avgR /= numtasks
		avgA /= numtasks

		file_name = 'results_reward_'+str(e)+'.npy'
		np.save(file_name, avgR)

		file_name = 'results_actions_'+str(e)+'.npy'
		np.save(file_name, avgA)

def load(numsteps, epislon):
	figs = []
	for e in epislon:
		avgR = np.load('results_'+str(e)+'.npy')
		regret = []
		sumr = 0
		for t in range(numsteps):
			sumr += avgR[t]
			regret.append(t*15 - sumr)
		fig = visualize(regret, e)
		figs.append(fig)
	add_legend_save(figs, epislon)


if __name__ == '__main__':
	# 10 arms 
	k = 10
	numsteps = 100000
	epislon = [0, 0.01, 0.1]
	run(k, numsteps, epislon)

	# load(numsteps, epislon)
	
		
	