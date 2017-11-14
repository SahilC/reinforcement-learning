import matplotlib.pyplot as plt

def add_legend_save(fig, epsilon, figure_name = 'esarsa_reward.png'):
	plt.legend(fig, [ 'Eps = ' + str(e) for e in epsilon])
	plt.savefig(figure_name)
	# plt.clf()


def init_plt():
	plt.figure(figsize=(30, 10))

def visualize(avgR, e):
	fig, = plt.plot(avgR)
	plt.ylabel('Reward')
	plt.xlabel('Steps')
	# plt.xscale('log')
	plt.xlim(xmin = 0, xmax = 10000)
	plt.ylim(ymin = -100, ymax = 100)
	# plt.show()
	# plt.savefig('mab_plot_exp3_'+str(e)+'regret.png')
	return fig
