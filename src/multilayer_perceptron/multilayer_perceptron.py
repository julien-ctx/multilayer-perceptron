import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utils import *

# A few interesting theories to determine the number of neurons in every hidden layer:
# The number of hidden neurons should be between the size of the input layer and the size of the output layer.
# The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
# The number of hidden neurons should be less than twice the size of the input layer.
# For this project, let use the mean of number of neurons in input layer, and number of neurons in output layer.

class MultilayerPerceptron:
	def __init__(self, df, algo_type):
		if algo_type not in algo.__members__:
			sys.exit(f"{color.RED}Error: wrong algorithm type specified.{color.END}")
		self.sample = df
		self.alpha = 0.001
		self.epochs = 800
		self.epsilon = 1e-8
		# Specifying values that will be used with different algorithms.
		self.algo_type = algo_type
		if self.algo_type in [algo.SGD.value, algo.NAG.value, algo.RMS.value]:
			self.batch_size = 96
		if self.algo_type == algo.NAG.value:
			# Usual value for momentum is 0.9. It is the momentum rate.
			# It determines the influence of the accumulated momentums on the current update.
			self.momentum = 0.9
		if self.algo_type == algo.RMS.value:
			self.decay_rate = 0.9
	
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

	def get_gradient(self, y_true, y_pred):
		# https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function
		# Gradient is different from logistic regression because the cross entropy loss function is not the same.
		# Add self.epsilon to avoid division by zero
		return -y_true / (y_pred + self.epsilon) + (1 - y_true) / (1 - y_pred + self.epsilon)

	# Activations is an array storing the output of each layer with feedforward.
	def get_activations(self, training, weights):
		# Input layer output is the raw input features (and the bias)
		activations = [training]
		for i in range(len(weights) - 1):
			activations.append(self.ReLU(activations[i] @ weights[i]))
		# Add the output layer activations (without bias this time)
		activations.append(self.softmax((activations[2] @ weights[2]).to_numpy()))
		return activations

	# https://www.youtube.com/watch?v=lglt8BZ6Ld4
	def backpropagation(self, y_true, activations, weights, training):
		# The following output delta formula is the simplified version of self.get_gradient(y_true, activations[-1]) * self.softmax_derivative(activations[-1])
		layers_delta = [activations[-1] - y_true]
		weights_it = len(weights) - 1
		for i in range(weights_it):
			layers_delta.append((layers_delta[i] @ weights[weights_it - i].T) * self.ReLU_derivative(activations[-weights_it - i]))
		for i in range(weights_it):
			gradient = (activations[-weights_it - i].T @ layers_delta[i]).to_numpy()
			if self.algo_type == algo.NAG.value:
				self.velocity[weights_it - i] = self.momentum * self.velocity[weights_it - i] - self.alpha * gradient
				weights[weights_it - i] += self.velocity[weights_it - i]
			elif self.algo_type == algo.RMS.value:
				self.cache[weights_it - i] = self.decay_rate * self.cache[weights_it - i] + (1 - self.decay_rate) * gradient ** 2
				weights[weights_it - i] = weights[weights_it - i] - (self.alpha * gradient) / (np.sqrt(self.cache[weights_it - i]) + self.epsilon)
			else:
				weights[weights_it - i] -= self.alpha * gradient
		gradient = (training.T @ layers_delta[-1]).to_numpy()
		if self.algo_type == algo.NAG.value:
			self.velocity[0] = self.momentum * self.velocity[0] - self.alpha * gradient
			weights[0] += self.velocity[0]
		elif self.algo_type == algo.RMS.value:
			self.cache[0] = self.decay_rate * self.cache[0] + (1 - self.decay_rate) * gradient ** 2
			weights[0] = weights[0] - (self.alpha * gradient) / (np.sqrt(self.cache[0]) + self.epsilon)
		else:
			weights[0] -= self.alpha * gradient
		return weights
	
	def print_loss(self, epoch, training_y_pred, validation_y_pred, training_diag, validation_diag):
		training_loss = self.loss(training_y_pred, training_diag)
		training_accuracy = self.accuracy(training_y_pred, training_diag)
		validation_loss = self.loss(validation_y_pred, validation_diag)
		validation_accuracy = self.accuracy(validation_y_pred, validation_diag)	
		print(f"{color.BOLD}Epoch {epoch}/{self.epochs} - Training Loss {training_loss} (Accuracy: {training_accuracy}%) - Validation Loss {validation_loss} (Accuracy: {validation_accuracy}%){color.END}")
		return training_loss, validation_loss

	def print_plots(self, total_epochs):
		fig, (ax1, ax2) = plt.subplots(1, 2)
		x = [x for x in range(total_epochs + 1)]
		ax1.plot(x, self.training_losses, color='blue', label='Training Loss Curve')
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Loss')
		ax1.set_title('Training Loss Curve')

		ax2.plot(x, self.validation_losses, color='orange', label='Validation Loss Curve')
		ax2.set_xlabel('Epochs')
		ax2.set_ylabel('Loss')
		ax2.set_title('Validation Loss Curve')

		plt.tight_layout()

		plt.show()

	def set_model_data(self):
		self.hidden_size = (self.sample.shape[1] + 2) // 2 + 1 # Add one for bias
		
		# Initialize weights
		np.random.seed(42)
		self.weights = [
			np.random.randn(self.sample.shape[1] + 1, self.hidden_size) * np.sqrt(1 / (self.sample.shape[1] + 1)),
			np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(1 / self.hidden_size),
			np.random.randn(self.hidden_size, 2) * np.sqrt(1 / self.hidden_size)
		]
		self.training_losses = []
		self.validation_losses = []
		if self.algo_type == algo.NAG.value:
			self.velocity = [np.zeros_like(weight) for weight in self.weights]
		elif self.algo_type == algo.RMS.value:
			self.cache = [np.zeros_like(weight) for weight in self.weights]

	def get_training_data(self):
		# Split the data into training and validation sets
		split_index = int(0.8 * len(self.sample))
		training = self.sample[:split_index]
		validation = self.sample[split_index:]
		
		# Convert the diagnosis values to one-hot encoded vectors
		training_diag = self.diagnosis[:split_index]
		training_diag = np.concatenate([training_diag == 1.0, training_diag == 0.0], axis=1).astype(float)
		validation_diag = self.diagnosis[split_index:]
		validation_diag = np.concatenate([validation_diag == 1.0, validation_diag == 0.0], axis=1).astype(float)

		# Add bias column to the training and validation sets
		training = self.add_bias(training)
		validation = self.add_bias(validation)
		return training, validation, training_diag, validation_diag

	def early_stopping(self, epoch):
		if len(self.validation_losses) > 1 and self.validation_losses[-2] <= self.validation_losses[-1]:
			if self.algo_type == algo.SGD.value and min(self.training_losses) * 2 < self.training_losses[-1]:
				return False
			print(f"{color.BLUE}Early stopping at epoch {epoch + 1}/{self.epochs} to avoid overfitting{color.END}")
			self.epochs = epoch
			return True
		return False

	def optimize(self, training, training_diag, validation, validation_diag, epoch):
		# Getting activations is computing the output activation of each layer using feedforward technique.
		training_activations = self.get_activations(training, self.weights)
		validation_activations = self.get_activations(validation, self.weights)    
		self.weights = self.backpropagation(training_diag, training_activations, self.weights, training)

		training_loss, validation_loss = self.print_loss(epoch + 1, training_activations[-1], validation_activations[-1], training_diag, validation_diag)

		self.training_losses.append(training_loss)
		self.validation_losses.append(validation_loss)

		return self.early_stopping(epoch)
		
	def fit(self):
		# Initialize data of the model: hidden layers size, weights...
		self.set_model_data()
		# Get training set, validation set, training diagnosis, validation diagnosis (true values)
		training, validation, training_diag, validation_diag = self.get_training_data()
		for epoch in range(self.epochs):
			if self.algo_type == algo.GD.value:
				if self.optimize(training, training_diag, validation, validation_diag, epoch):
					break
			elif self.algo_type in [algo.SGD.value, algo.NAG.value, algo.RMS.value]:
				batch = training.sample(n=self.batch_size)
				diagnosis = training_diag[batch.index]
				if self.optimize(batch, diagnosis, validation, validation_diag, epoch):
					break
	
		self.weights = np.array(self.weights, dtype=object)
		np.save('../../assets/weights.npy', self.weights)
		self.print_plots(self.epochs)

	# Accuracy score, as a percentage, inspired from scikit-learn.
	def accuracy(self, y_pred, y_true):
		y_pred = np.where(y_pred == np.max(y_pred, axis=1)[:, np.newaxis], 1.0, y_pred)
		y_pred = np.where(y_pred == np.min(y_pred, axis=1)[:, np.newaxis], 0.0, y_pred)
		correct_predictions = np.sum(np.equal(y_pred, y_true).all(axis=1))
		return np.round(correct_predictions * 100 / y_pred.shape[0], 2)

	def loss(self, y_pred, y_true):
		# Log function is undefined for 0 and < 0 values. Therefore we add self.epsilon to avoid endless values.
		return np.round(-np.mean(y_true * np.log(y_pred + self.epsilon) + (1 - y_true) * np.log(1 - y_pred + self.epsilon)), 4)
	
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
