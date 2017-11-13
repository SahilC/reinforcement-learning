import matplotlib.pyplot as plt

def add_legend_save(fig, epsilon, figure_name = 'mab_plot_mab_regret.png'):
	plt.legend(fig, epsilon)
	plt.savefig(figure_name)
	# plt.clf()

def visualize(avgR, e):
	fig, = plt.plot(avgR)
	plt.ylabel('PseudoRegret')
	plt.xlabel('Steps')
	# plt.xscale('log')
	plt.xlim(0)
	# plt.show()
	# plt.savefig('mab_plot_exp3_'+str(e)+'regret.png')
	return fig

def visualize_actions(avgA, e):
	fig,  = plt.plot(avgA)
	plt.ylabel('Average Number of times Pulled')
	plt.xlabel('Arm')
	# plt.xscale('log')
	plt.xlim(-5)
	return fig
	# plt.show()
