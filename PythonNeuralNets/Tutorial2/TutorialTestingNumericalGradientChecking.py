from tutorial import NeuralNetwork
import numpy as np

# General idea behind numerical gradient checking

def f(x):
	return x**2

epsilon = 10**-4
x = 1.5

for _ in range(50):
	# Gradient between (x-e, x+e)
	numericGradient = (f(x+epsilon) - f(x-epsilon))/(2*epsilon)
	#print(numericGradient, 2*x)
	x -= epsilon * numericGradient * 1000


# Neural Network Testing

def computeNumericalGradient(N, X, Y):
	paramsInitial = N.getParams()
	numgrad = np.zeros(paramsInitial.shape)
	perturb = np.zeros(paramsInitial.shape)
	e = 10**-4

	for p in range(len(paramsInitial)):
		# Set perturbation vector
		perturb[p] = e
		N.setParams(paramsInitial + perturb)
		loss2 = N.costFunction(X, Y)

		N.setParams(paramsInitial - perturb)
		loss1 = N.costFunction(X, Y)

		# Compute Numerical Gradient
		numgrad[p] = (loss2 - loss1) / (2*e)

		# Return the value we changed back to zero
		perturb[p] = 0

	# Return Params to original value
	N.setParams(paramsInitial)

	return numgrad


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
grad = NN.computeGradients(X, Y) 
numgrad = computeNumericalGradient(NN, X, Y)

normDifferenceComparison = np.linalg.norm(grad-numgrad) / np.linalg.norm(grad+numgrad)

print("Calculus Computed Gradients:")
print( grad	)
print()
print("Numerically Computed Gradients:")
print( numgrad )
print()
print("Difference:")
print( grad - numgrad )
print()
print("Norm Difference:")
print( normDifferenceComparison )
print()

