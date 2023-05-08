import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import KFold
import sys
sys.path.append('..')
from utils import color

# A few interesting theories to determine the number of neurons in every hidden layer:
# The number of hidden neurons should be between the size of the input layer and the size of the output layer.
# The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
# The number of hidden neurons should be less than twice the size of the input layer.
# For this project, let use the mean of number of neurons in input layer, and number of neurons in output layer.

class MultilayerPerceptron:
	def __init__(self, df):
		self.sample = df
		self.alpha = 0.01
		self.epochs = 1000
	
	def drop_irrelevant_data(self):
		# Thanks to histogram, we can see that Feature 15 (and 12?) has almost the same distribution independently from the type of tumor.
		# It may create noise in the model, and it can be a good idea to remove it.
		# Instead of removing it by hand, we can remove then using Kolmogorov test.
		# If pvalue 0.05, we can exclude this hypothesis that samples distributions are similar.
		# https://fr.wikipedia.org/wiki/Test_de_Kolmogorov-Smirnov

		features_to_drop = []
		sample = self.sample
		for i, feature in enumerate(self.sample.drop('Diagnosis', axis=1).columns):
			statistic, pvalue = ks_2samp(self.sample[self.sample.Diagnosis != 1.0][feature], self.sample[self.sample.Diagnosis != 0.0][feature])
			if pvalue > 0.05:
				features_to_drop.append(feature)
				sample = sample.drop(feature, axis=1)
		self.sample = sample
  
		# Separate the diagnosis from the input layer.
		self.diagnosis = self.sample.Diagnosis.to_numpy().reshape(-1, 1)
		self.sample = self.sample.drop('Diagnosis', axis=1)
		np.save('../../assets/features.npy', features_to_drop)

	def add_bias(self, array):
		# Copy to let pandas understand I don't want to edit the original df.
		array = array.copy()
		array['Bias'] = np.ones(array.shape[0])
		return array

	def standardize(self):
		self.sample = self.sample.apply(lambda x : (x - np.mean(x)) / np.std(x))

	# Activations is an array storing the output of each layer.
	def get_activations(self, training, weights):
		# Input layer output is the raw input features (and the bias)
		activations = [training]
		for i in range(len(weights) - 1):
			activations.append(self.ReLU(activations[i] @ weights[i]))
		# Add the output layer activations (without bias this time)
		activations.append(self.softmax((activations[2] @ weights[2]).to_numpy()))
		return activations
		
	def get_gradient(self, y_true, y_pred):
		# https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function
		# Gradient is different from logistic regression because the cross entropy loss function is not the same.
		# Add epsilon to avoid division by zero
		epsilon = 1e-8
		return -y_true / (y_pred + epsilon) + (1 - y_true) / (1 - y_pred + epsilon)

	# https://www.youtube.com/watch?v=lglt8BZ6Ld4
	def backpropagation(self, y_true, activations, weights, training):
		# The following output delta formula is the simplified version of self.get_gradient(y_true, activations[-1]) * self.softmax_derivative(activations[-1])
		output_delta = activations[-1] - y_true
		second_hidden_layer_delta = (output_delta @ weights[2].T) * self.ReLU_derivative(activations[-2])
		first_hidden_layer_delta = (second_hidden_layer_delta @ weights[1].T)  * self.ReLU_derivative(activations[-3])
		weights[2] -= (self.alpha * (activations[-2].T @ output_delta)).to_numpy()
		weights[1] -= (self.alpha * (activations[-3].T @ second_hidden_layer_delta)).to_numpy()
		weights[0] -= (self.alpha * (training.T @ first_hidden_layer_delta)).to_numpy()
		return weights
	
	def print_loss(self, i, epoch, training_y_pred, validation_y_pred, training_diag, validation_diag):
		training_loss = self.loss(training_y_pred, training_diag)
		training_accuracy = self.accuracy(training_y_pred, training_diag)
		validation_loss = self.loss(validation_y_pred, validation_diag)
		validation_accuracy = self.accuracy(validation_y_pred, validation_diag)	
		if epoch == self.epochs:
			print(f"{color.GREEN}Fold {i}/10 - Epoch {epoch}/{self.epochs} - Training Loss {training_loss} (Accuracy: {training_accuracy}%) - Validation Loss {validation_loss} (Accuracy: {validation_accuracy}%){color.END}")
		elif epoch < 20 or epoch % 10 == 0:
			print(f"{color.BOLD}Fold {i}/10 - Epoch {epoch}/{self.epochs} - Training Loss {training_loss} (Accuracy: {training_accuracy}%) - Validation Loss {validation_loss} (Accuracy: {validation_accuracy}%){color.END}")

	def fit(self):
		self.hidden_size = 12#(self.sample.shape[1] + 2) // 2 + 1 # + 1 for bias
		self.weights = []
		
		# Generate folds to follow the subject guidelines.
		kf = KFold(n_splits=10)
		# training_i and validation_i are indices for data in self.sample for each fold.
		for i, (training_i, validation_i) in enumerate(kf.split(self.sample)):
			# Gives a seed to numpy to understand that random values will always be the same after running the program several times.
			np.random.seed(42 + i)
			# https://github.com/christianversloot/machine-learning-articles/blob/main/he-xavier-initialization-activation-functions-choose-wisely.md
			# https://cs230.stanford.edu/section/4/
			weights = [
				np.random.randn(self.sample.shape[1] + 1, self.hidden_size) * np.sqrt(1 / (self.sample.shape[1] + 1)),
				np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(1 / self.hidden_size),
				np.random.randn(self.hidden_size, 2) * np.sqrt(1 / self.hidden_size)
			]
			# Get training and validation batches.
			training = self.sample.iloc[training_i, :]
			training = self.add_bias(training)
			training_diag = self.diagnosis[training_i, :]
			training_diag = np.concatenate([training_diag == 1.0, training_diag == 0.0], axis=1).astype(float)
			validation = self.sample.iloc[validation_i, :]
			validation = self.add_bias(validation)
			validation_diag = self.diagnosis[validation_i, :]
			validation_diag = np.concatenate([validation_diag == 1.0, validation_diag == 0.0], axis=1).astype(float)
			for epoch in range(self.epochs):
				# Getting activations is computing the output of each layer using feedforward technique.
				training_activations = self.get_activations(training, weights)
				validation_activations = self.get_activations(validation, weights)	
				self.print_loss(i + 1, epoch + 1, training_activations[-1], validation_activations[-1], training_diag, validation_diag)
				weights = self.backpropagation(training_diag, training_activations, weights, training)
			self.weights.append(weights)
			print()
			break
		means = []
		for i in range(3):
			means.append(np.mean([weight[i] for weight in self.weights], axis=0))
		means = np.array(means, dtype=object)
		np.save('../../assets/weights.npy', means)

	# Accuracy score, as a percentage, inspired from scikit learn.
	def accuracy(self, y_pred, y_true):
		y_pred = np.where(y_pred == np.max(y_pred, axis=1)[:, np.newaxis], 1.0, y_pred)
		y_pred = np.where(y_pred == np.min(y_pred, axis=1)[:, np.newaxis], 0.0, y_pred)
		correct_predictions = np.sum(np.equal(y_pred, y_true).all(axis=1))
		return np.round(correct_predictions * 100 / y_pred.shape[0], 2)

	def loss(self, y_pred, y_true):
		# Log function is undefined for 0 and < 0 values. Therefore we add epsilon to avoid endless values.
		epsilon = 1e-8
		return np.round(-np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)), 4)
	
	# Softmax is used by neurons of output layer.
	def softmax(self, z):
		# z are the logits of the neurons, ie the output before activation.
		exp = np.exp(z)
		return exp / np.sum(exp, axis=1, keepdims=True)

	# Rectified Linear Unit function replaces negative values by 0.
	# It is used by neurons of hidden layers.
	def ReLU(self, z):
		return np.maximum(0, z)

	# https://deepnotes.io/softmax-crossentropy
	def softmax_derivative(self, activations):
		return activations * (1 - activations)

	def ReLU_derivative(self, activations):
		return (activations > 0) * 1
