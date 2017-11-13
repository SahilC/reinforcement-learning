import numpy as np
from exp3 import exp3
from greedy_bandit import *
from ucb import *
from visualize import *
import os

def run(k, numsteps, epislon):
	numtasks = 200
	
	for e in epislon:
		print 'Starting'

		avgR = np.zeros((numsteps, ))
		
		avgA = np.zeros((k, ))

		for task in range(numtasks):
			# bandit_task = exp3(k, e)
		    bandit_task = greedy_action_selection(k, numsteps, e)
		    # bandit_task = ucb_action_selection(k, numsteps, e)
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
		avgR = np.load('results_reward_'+str(e)+'.npy')
		avgA = np.load('results_actions_'+str(e)+'.npy')
		regret = []
		sumr = 0
		for t in range(numsteps):
			#a = avgA[t]
			sumr += avgR[t]
			#print 15*t
			#print np.sum(avgA*np.array([(5+i) for i in xrange(1,11)])*t/numsteps)
			regret.append(15*t - np.sum(avgA*np.array([(5+i) for i in xrange(1,11)])*t/numsteps))
		fig = visualize(regret, e)
		figs.append(fig)
	add_legend_save(figs, epislon,'mab_plot_exp3_regret.png')

def load_pulled(k, epislon):
	figs = []
	for e in epislon:
		avgA = np.load('results_actions_'+str(e)+'.npy')
		# print avgA.shape
		# print avgA
		fig = visualize_actions(avgA, e)
		figs.append(fig)
	add_legend_save(figs, epislon, 'mab_plot_exp3_actions.png')


if __name__ == '__main__':
	# 10 arms 
	k = 10
	numsteps = 100000
	epislon = [0, 0.01, 1, 2]
	# epislon = [0, 0.01, 0.1]
	#epislon = [100000, 10000, 1000]
	# run(k, numsteps, epislon)

	# load(numsteps, epislon)
	load_pulled(numsteps, epislon)
	
		
	
