import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from utils import get_df
import seaborn as sns

if __name__ == '__main__':
	df = get_df()
	# Get correlation matrix
	# -1 indicates strong negative correlation
	# 1 indicates strong positive correlation
	# 0 indicates no correlation
	corr = df.corr()

	sns.heatmap(corr, cmap='coolwarm')

	plt.get_current_fig_manager().set_window_title('Features correlation')
	plt.show()
