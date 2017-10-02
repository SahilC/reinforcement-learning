import matplotlib.pyplot as plt

def add_legend_save(fig, epsilon, figure_name = 'mab_plot_mab_regret.png'):
	plt.legend(fig, epsilon)
	plt.savefig(figure_name)
	plt.clf()

def visualize(avgR, e):
	fig,  = plt.plot(avgR)
	plt.ylabel('Average reward')
	plt.xlabel('Steps')
	# plt.xscale('log')
	plt.xlim(0)
	return fig
	# plt.show()

def visualize_actions(avgA, e):
	fig,  = plt.plot(avgA)
	plt.ylabel('Average Number of times Pulled')
	plt.xlabel('Arm')
	# plt.xscale('log')
	plt.xlim(-5)
	return fig
	# plt.show()
