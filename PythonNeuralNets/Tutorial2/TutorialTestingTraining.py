from tutorial import NeuralNetwork

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

class trainer:
	def __init__(self, N):
		# Make local reference to Neural Network
		self.N = N

	def costFunctionWrapper(self, params, X, Y):
		self.N.setParams(params)
		cost = self.N.costFunction(X, Y)
		grad = self.N.computeGradients(X, Y)
		return cost, grad

	def callBackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.X, self.Y))

	def train(self, X, Y):

		# Set internal variables for callback function
		self.X = X
		self.Y = Y

		# Make empty list to store costs
		self.J = []


		# We are using the BFGS gradient descent algorithm
		# Passed requirementss:
		#	- Function that reqiures a single vector of parameters
		#	- Input and Output data
		# Returns costs and gradients
		params0 = self.N.getParams()
		_res = optimize.minimize(
						self.costFunctionWrapper, 
						params0, 
						jac = True, 
						method = 'BFGS', 
						args = (X,Y),
						options = { 'maxiter':200, 'disp':True },
						callback = self.callBackF
					)

		# Set params of neural network
		self.N.setParams(_res.x)
		self.optimisationResults = _res

# Hours slept 
# Hours studying
X = np.array(
	([3,5],[5,1],[10,2]),
	dtype=float)

# Test Score [0,100]
Y = np.array(
	([75], [82], [93]),
	dtype=float)

# Scaling to between 0 and 1
X = X/np.max(X, axis=0)
Y = Y/100

NN = NeuralNetwork()
T = trainer(NN)
T.train(X, Y)


# Plot the results
plt.plot(T.J)
plt.grid(1)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()


# Test outputs
yHat = NN.forward(X)

print("Prediction:")
print(yHat)
print()
print("Expected:")
print(Y)
print()

##
## Probing at different combinations of sleep/study
##

# Make plots
hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)
# Normalise data
hoursSleepNorm = hoursSleep/10
hoursStudyNorm = hoursStudy/5
# Create 2D version of input for plotting
a, b = np.meshgrid(hoursSleepNorm, hoursStudyNorm)
# Join into single input matrix
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()
# Get outputs
allOutputs = NN.forward(allInputs)

# Make Contour plot
yy = np.dot(hoursStudy.reshape(100, 1), np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100, 1), np.ones((1,100))).T
CS = plt.contour(xx, yy, 100*allOutputs.reshape(100,100))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel("Hours sleep")
plt.ylabel("Hours Study")
plt.show()


# Note: that we now have a NN that is subject to overfitting
# Note: that outputs are unpredictable in unknown regions

