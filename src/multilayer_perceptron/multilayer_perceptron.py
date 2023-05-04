import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import KFold

# A few interesting theories to determine the number of neurons in every hidden layer:
# The number of hidden neurons should be between the size of the input layer and the size of the output layer.
# The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
# The number of hidden neurons should be less than twice the size of the input layer.
# For this project, let use the mean of number of neurons in input layer, and number of neurons in output layer.

class MultilayerPerceptron:
	def __init__(self, df):
		self.sample = df
		self.alpha = 0.001
	
	def drop_irrelevant_data(self):
		# Thanks to histogram, we can see that Feature 15 (and 12?) has almost the same distribution independently from the type of tumor.
		# It may create noise in the model, and it can be a good idea to remove it.
		# Instead of removing it by hand, we can remove then using Kolmogorov test.
		# If pvalue 0.05, we can exclude this hypothesis that samples distributions are similar.
		# https://fr.wikipedia.org/wiki/Test_de_Kolmogorov-Smirnov

		sample = self.sample
		for i, feature in enumerate(self.sample.drop('Diagnosis', axis=1).columns):
			statistic, pvalue = ks_2samp(self.sample[self.sample.Diagnosis != 1.0][feature], self.sample[self.sample.Diagnosis != 0.0][feature])
			if pvalue > 0.05:
				sample = sample.drop(feature, axis=1)
				
		self.sample = sample
  
		# Separate the diagnosis from the input layer.
		self.diagnosis = self.sample.Diagnosis.to_numpy().reshape(-1, 1)
		self.sample = self.sample.drop('Diagnosis', axis=1)

	def add_bias(self, array):
		# Copy to let pandas understand I don't want to edit the original df.
		array = array.copy()
		array['Bias'] = np.ones(array.shape[0])
		return array

	def standardize(self):
		self.sample = self.sample.apply(lambda x : (x - np.mean(x)) / np.std(x))

	# Activations is an array storing the output of each layer.
	def get_activations(self, training):
		# Input layer output is the raw input features (and the bias)
		activations = [self.add_bias(training)]
		for i in range(len(self.weights) - 1):
			activations.append(self.ReLU(activations[i] @ self.weights[i]))
		# Add the output layer activations (without bias this time)
		activations.append(self.softmax((activations[2] @ self.weights[2]).to_numpy()))
		print(activations[-1])
		exit()
  
	def fit(self):
		self.hidden_size = (self.sample.shape[1] + 2) // 2 + 1 # + 1 for bias

		# Gives a seed to numpy to understand that random values will always be the same after running the program several times.
		np.random.seed(42)
		# https://github.com/christianversloot/machine-learning-articles/blob/main/he-xavier-initialization-activation-functions-choose-wisely.md
		# https://cs230.stanford.edu/section/4/
		self.weights = [
			np.random.randn(self.sample.shape[1] + 1, self.hidden_size),
			np.random.randn(self.hidden_size, self.hidden_size),
			np.random.randn(self.hidden_size, 2)
		]

		# Generate folds to follow the subject guidelines.
		kf = KFold(n_splits=10)
		# training_i and validation_i are indices for data in self.sample for each fold.
		for training_i, validation_i in kf.split(self.sample):
			training = self.sample.iloc[training_i, :]
			validation = self.sample.iloc[validation_i, :]
			for epoch in range(1000):
				activations = self.get_activations(training)
				# self.weights -= self.alpha * get_gradient()
	
	# Softmax is used by neurons of output layer.
	def softmax(self, z):
		# z are the logits of the neurons, ie the output before activation.
		exp = np.exp(z)
		return exp / np.sum(exp, axis=1, keepdims=True)

	# Rectified Linear Unit function replaces negative values by 0.
	# It is used by neurons of hidden layers.
	def ReLU(self, z):
		return np.maximum(0, z)
