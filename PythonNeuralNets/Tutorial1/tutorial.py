'''
https://medium.freecodecamp.org/building-a-3-layer-neural-network-from-scratch-99239c4af5d3

The input layer (x) consists of 178 neurons.
A1, the first layer, consists of 8 neurons.
A2, the second layer, consists of 5 neurons.
A3, the third and output layer, consists of 3 neurons.

'''

#
# Import Libraries
#

# Libraries
import pandas as pd
import numpy as np

# Dataset
df = pd.read_csv('../input/W1data.csv')
df.head()

# Maths plotting
import matplotlib
import matplotlib.pyplot as plt

# SciKitLearn ML library
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

#
# Initialisation
#

np.random.seed(0)

#
# Forward Propagation
#

def forward_prop(model, a0):

	# Load parameters: layer intersections 1,2,3
	W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']

	# First linear step
	z1 = a0.dot(W1) + b1
	# First activation function
	a1 = np.tanh(z1)

	# Second linear step
	z2 = a0.dot(W2) + b2
	# Second activation function
	a2 = np.tanh(z2)
	
	# Third linear step
	z3 = a0.dot(W3) + b3
	# Third activation function: softmax for probabilities
	a3 = softmax(z3)


#
# Backwards Propagation
#

# This is the backward propagation function
def backward_prop(model,cache,y):
	# Load parameters from model
	W1,	b1,	W2,	b2,	W3,	b3 = model['W1'], model['b1'], model['W2'],	model['b2'],model['W3'],model['b3']
	
	# Load forward propagation results
	a0,a1, a2,a3 = cache['a0'],cache['a1'],cache['a2'],cache['a3']
	
	# Get number of	samples
	m =	y.shape[0]
	
	# Calculate	loss derivative	with respect to	output
	dz3	= loss_derivative(y=y,y_hat=a3)
	
	# Calculate	loss derivative	with respect to	second layer weights
	dW3	= 1/m*(a2.T).dot(dz3) #dW2 = 1/m*(a1.T).dot(dz2) 
	
	# Calculate	loss derivative	with respect to	second layer bias
	db3	= 1/m*np.sum(dz3, axis=0)
	
	# Calculate	loss derivative	with respect to	first layer
	dz2	= np.multiply(dz3.dot(W3.T)	,tanh_derivative(a2))
	
	# Calculate	loss derivative	with respect to	first layer	weights
	dW2	= 1/m*np.dot(a1.T, dz2)
	
	# Calculate	loss derivative	with respect to	first layer	bias
	db2	= 1/m*np.sum(dz2, axis=0)
	
	dz1	= np.multiply(dz2.dot(W2.T),tanh_derivative(a1))
	
	dW1	= 1/m*np.dot(a0.T,dz1)
	
	db1	= 1/m*np.sum(dz1,axis=0)
	
	# Store	gradients
	grads =	{'dW3':dW3,	'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
	return grads
