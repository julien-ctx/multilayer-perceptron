import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')
from utils import get_df

if __name__ == '__main__':
	df = get_df()
	
	# Plot the histograms
	fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(16, 9))
	for i, feature in enumerate(df.drop('Diagnosis', axis=1).columns):
		axs[i // 4][i % 4].hist(df[df.Diagnosis != 1][feature], bins=20, alpha=0.5, color='blue')
		axs[i // 4][i % 4].hist(df[df.Diagnosis != 0][feature], bins=20, alpha=0.5, color='red')
		axs[i // 4][i % 4].set_title(feature, fontsize=10)

	# Add space between histograms
	plt.subplots_adjust(hspace=1, wspace=0.3)
	# Remove unused histogram
	axs[7, 3].axis('off')
	axs[7, 2].axis('off')
	# Set window title
	plt.get_current_fig_manager().set_window_title('Malignant VS Benign')

	plt.savefig('../../assets/histogram.png')
