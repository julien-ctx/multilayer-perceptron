import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

class MultilayerPerceptron:
	def __init__(self, df):
		self.sample = df
	
	def drop_irrelevant_data(self):
		# Thanks to histogram, we can see that Feature 16 (and 13?) has almost the same distribution independently from the type of tumor.
		# It may create noise in the model, and it can be a good idea to remove it.
		# Instead of removing it by hand, we can remove then using Kolmogorov test.
		# If pvalue 0.05, we can exclude this hypothesis that samples distributions are similar.
		# https://fr.wikipedia.org/wiki/Test_de_Kolmogorov-Smirnov

		sample = self.sample
		for i, feature in enumerate(self.sample.drop('Malignant', axis=1).columns):
			statistic, pvalue = ks_2samp(self.sample[self.sample.Malignant != 1.0][feature], self.sample[self.sample.Malignant != 0.0][feature])
			if pvalue > 0.05:
				sample = sample.drop(feature, axis=1)
				
		self.sample = sample
 
	def standardize(self):
		self.sample = self.sample.apply(lambda x : (x - np.mean(x)) / np.std(x))
