import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# 357 are benign
# 211 are malignant

if __name__ == '__main__':
	if not os.path.exists('../../assets/data.csv'):
		sys.exit("Error: data.csv doesn't exist")
	df = pd.read_csv('../../assets/data.csv')
	df.columns = [f"Feature {i + 1}" if i < 1 else f"Feature {i}" if i > 1 else "Malignant" for i in range(len(df.columns))]

	fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(16, 10))

	for i, feature in enumerate(df.drop('Malignant', axis=1).columns):
		axs[i // 4][i % 4].hist(df[df.Malignant != 'M'][feature], bins=20, alpha=0.5, color='blue')
		axs[i // 4][i % 4].hist(df[df.Malignant != 'B'][feature], bins=20, alpha=0.5, color='red')
		axs[i // 4][i % 4].set_title(feature, fontsize=10)

	plt.subplots_adjust(hspace=1, wspace=0.3)
	axs[7, 3].axis('off')
	plt.show()
