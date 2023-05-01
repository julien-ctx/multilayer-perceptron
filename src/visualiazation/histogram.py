import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from utils import get_df

# 357 are benign
# 211 are malignant

if __name__ == '__main__':
	df = get_df()

	fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(16, 9))

	for i, feature in enumerate(df.drop('Malignant', axis=1).columns):
		axs[i // 4][i % 4].hist(df[df.Malignant != 1][feature], bins=20, alpha=0.5, color='blue')
		axs[i // 4][i % 4].hist(df[df.Malignant != 0][feature], bins=20, alpha=0.5, color='red')
		axs[i // 4][i % 4].set_title(feature, fontsize=10)

	# Add space between histograms
	plt.subplots_adjust(hspace=1, wspace=0.3)
	# Remove unused histogram
	axs[7, 3].axis('off')
	# Set window title
	plt.get_current_fig_manager().set_window_title('Malignant VS Benign')

	plt.show()
