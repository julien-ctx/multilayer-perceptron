import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from utils import get_df

if __name__ == '__main__':
	df = get_df()

	fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(16, 9))
	# Change color properties
	boxprops = dict(linewidth=1.5, color='r')
	whiskerprops = dict(linewidth=1.5, color='b')

	for i, feature in enumerate(df.drop('Malignant', axis=1).columns):
		axs[i // 4][i % 4].boxplot([df[df.Malignant != 0][feature], df[df.Malignant != 1][feature]], boxprops=boxprops, whiskerprops=whiskerprops)
		axs[i // 4][i % 4].set_title(feature, fontsize=10)

	plt.subplots_adjust(hspace=1, wspace=0.3)
	axs[7, 3].axis('off')
	# Set window title
	plt.get_current_fig_manager().set_window_title('Malignant VS Benign')

	plt.show()
