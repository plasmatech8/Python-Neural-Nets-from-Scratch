'''
Welsh Labs Tutorial: Neural Networks Demystified.
https://www.youtube.com/watch?v=bxe2T-V8XRs

'''

import numpy as np

##
## Neural Network
##
class NeuralNetwork:
	def __init__(self):
		# Define Hyperparameters
		self.inputLayerSize = 2
		self.hiddenLayerSize = 3
		self.outputLayerSize = 1

		# Weights
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

	##
	## USAGE
	##

	def forward(self, X):
		# Propagate inputs through the network
		# 1) z^(2) = XW^(1)			... Collect sum of activation outputs from input/previous layer
		# 2) a^(2) = f(z^(2))		... Pass sum through activation function
		# 3) z^(3) = a^(2)W^(2)		... Collect sum of activation outputs from previous layer
		# 4) y_out = f(z^(3))		... Pass sum through activation function
		z2 = np.dot(X, self.W1)
		a2 = NeuralNetwork.sigmoid(z2)
		z3 = np.dot(a2, self.W2)
		yHat = NeuralNetwork.sigmoid(z3)
		self.z2 = z2
		self.a2 = a2
		self.z3 = z3
		self.yHat = yHat
		return yHat
	
	##
	## COST FUNCTIONS
	##

	def costFunction(self, X, Y):
		self.yHat = self.forward(X)
		J = 0.5*sum((Y-self.yHat)**2)
		return J

	def costFunctionPrime(self, X, Y):
		# Compute derivative with respect to W1 and W2
		self.forward(X)
		delta3 = np.multiply(-(Y-self.yHat), NeuralNetwork.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)
		delta2 = np.dot(delta3, self.W2.T) * NeuralNetwork.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)
		return dJdW1, dJdW2

	##
	## ACTIVATION FUNCTIONS
	##

	def sigmoid(z):
		return 1/(1 + np.exp(-z))

	def sigmoidPrime(z):
		return np.exp(-z)/((1+np.exp(-z))**2)

	##
	## HELPER FUNCTIONS
	##

	def getParams(self):
		# Get W1 and W2 rolled into a single vector
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params

	def setParams(self, params):
		# Set W1 and W2 using a single vector
		W1_start = 0

		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(
			params[W1_start:W1_end],
			(self.inputLayerSize, self.hiddenLayerSize)
			)

		W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
		self.W2 = np.reshape(
			params[W1_end:W2_end],
			(self.hiddenLayerSize, self.outputLayerSize)
			)

	def computeGradients(self, X, Y):
		dJdW1, dJdW2 = self.costFunctionPrime(X, Y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))



