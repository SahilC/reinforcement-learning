import matplotlib.pyplot as plt

def add_legend_save(fig, epsilon):
	plt.legend(fig, epsilon)
	plt.savefig('mab_plot_log_regret.png')

def visualize(avgR, e):
	fig,  = plt.plot(avgR)
	plt.ylabel('Average reward')
	plt.xlabel('Steps')
	# plt.xscale('log')
	plt.xlim(-5)
	return fig
	# plt.show()

