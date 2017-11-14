from qlearn import QLearn
import numpy as np

class grid_world:
	def __init__(self, rows, cols , agent_state, start_state, end_state, bad_state):
		self.rows = rows
		self.cols = cols
		self.start_state = start_state
		self.bad_state = bad_state
		self.end_state = end_state

		self.grid = [[-1 for j in xrange(cols)] for i in xrange(rows)]

		self.grid[start_state[0]][start_state[1]] = 0
		self.grid[end_state[0]][end_state[1]] = 100
		self.grid[bad_state[0]][bad_state[1]] = -70

		self.agent_state = agent_state

	def print_state(self):
		for i in xrange(self.rows):
			for j in xrange(self.cols):
				print '%3d'%self.grid[i][j],
			print ''

	def get_action_state(self, state, a):
		new_state = [state[0],state[1]]
		if a == 0 and new_state[0] < (self.rows -1):
			new_state[0] += 1
		elif a == 1 and new_state[0] > 0:
			new_state[0] -= 1
		elif a == 2 and new_state[1] < (self.cols - 1):
			new_state[1] += 1
		elif a == 3 and new_state[1] > 0:
			new_state[1] -= 1
		return (new_state[0],new_state[1])

	def update_agent_state(self, q):
		# print(self.agent_state)
		rewards = []
		states = [self.agent_state]
		while self.agent_state[0] != self.end_state[0] or self.agent_state[1] != self.end_state[1]:
			action = q.chooseAction(self.agent_state)
			old_state = self.agent_state
			self.agent_state = self.get_action_state(self.agent_state, action)
			reward = self.grid[self.agent_state[0]][self.agent_state[1]]
			# print(action)
			# print(self.agent_state)
			# print(reward)
			# print('---------------')
			rewards.append(reward)
			states.append(self.agent_state)
			q.learn(old_state, action, reward, self.agent_state)
		print states
		return np.sum(rewards)

if __name__ == '__main__':
	actions  = [0, 1 , 2, 3]
	q = QLearn(actions)
	rewards = []
	print('Started')
	for i in xrange(10000):
		g = grid_world(4,4,(1,0),(1,0),(3,3),(2,1))
		r = g.update_agent_state(q)
		rewards.append(r)

		print i, r
		# q.printQ()
		# q.printV()

