import numpy as np

def sigmoid(z):
   '''
   The sigmoid activation function
   
   Arguments:
   z -- vector of dimension m*n
   
   Returns:
   sigmoid of individual values in vector z in same dimension
   '''
   return (1/(1 + np.power(np.exp(z),(-1))))
   
def initialize_parameters(n_x, network_dimensions):
   '''
   Function to initialize parameters

   Arguments:
   n_x -- dimension of input
   network_dimensions -- dimension of hidden layers and output (total L)

   Returns:
   parameters -- initialized parameters W1, b1, ... WL, bL
   '''
   parameters = {}
   parameters['W1'] = np.random.randn(network_dimensions[0], n_x.shape[0]) * 0.01
   parameters['b1'] = np.zeros((network_dimensions[0], 1))
   for l in range(len(network_dimensions)-1):
      parameters['W' + str(l+2)] = np.random.randn(network_dimensions[l+1], network_dimensions[l]) * 0.01
      parameters['b' + str(l+2)] = np.zeros((network_dimensions[l+1], 1))
   return parameters

def forward_propogation(x, parameters, L):
   '''
   Function to implement forward propogation of neural network
   
   Arguments:
   x -- initial input for neural network
   parameters -- initialized parameters for neural network (using initialize_parameters function)
   activations -- cache for storing activations value
   L -- number of layers of neural network
   
   Returns:
   AL -- final value determined from forward propogation
   activations -- caching weigths W, biases b and intermediate values Z (linear activation) for backward propogation
   '''
   activations = {}
   activations['A0'] = x
   prev_A = x
   #ReLU activation for L-1 layers
   for l in range(1, L):
      activations['Z' + str(l)] = np.dot(parameters['W' + str(l)], prev_A) + parameters['b' + str(l)]
      activations['A' + str(l)] = np.maximum(np.zeros(activations['Z' + str(l)].shape), activations['Z' + str(l)]) #ReLU activation
      prev_A = activations['A' + str(l)]
     
   #sigmoid activation for last layer (binary classification
   activations['Z' + str(L)] = np.dot(parameters['W' + str(L)], prev_A) + parameters['b' + str(L)]
   activations['A' + str(L)] = sigmoid(activations['Z' + str(L)]) #ReLU activation
   return activations['A' + str(L)], activations
   
def cost_function(AL, Y):
   '''
   Cost function for sigmoid activation function - J = -(1/m)(Y log(AL) + (1-Y) log(1-AL))
   
   Arguments:
   AL -- output from forward propogation
   Y -- actual output of training data
   
   Returns:
   J -- value of the cost function
   '''
   m = Y.shape[1]
   J = -(1/m) * (np.sum(np.multiply(Y, np.log(AL)) - np.multiply((1-Y), np.log(1-AL))))
   return J
   
def relu_backward(dA, z):
   '''
   Compute dZ. Gradient of ReLU is 0 if value is <= 0 else value. dZ = dA * g'(z)
   
   Arguments:
   dA -- The base value to be considered
   z -- value to be queried
   
   Returns:
   dZ -- The ReLU backward value
   '''
   dZ = np.array(dA, copy = True) # A and Z are same. So only need to copy and filter for 0
   dZ[z <= 0] = 0 #set value to 0 if z is 0
   return dZ
   
def sigmoid_backward(dA, z):
   '''
   Compute dZ. Gradient of sigmoid. dZ = dA * g'(z) = dA * s * (1-s) where s is the sigmoid of Z
   
   Arguments:
   dA -- The base value to be considered
   z -- value to be queried
   
   Returns:
   dZ -- The sigmoid backward value
   '''
   s = sigmoid(z)
   dZ = dA * s * (1-s)
   return dZ
   
def backward_propogation(Y, AL, parameters, activations, L):
   '''
   Function to implement backward propogation of neural network
   
   Arguments:
   Y -- actual output for neural network
   AL -- output predicted from feedforward pass
   parameters -- initialized parameters for neural network (using initialize_parameters function)
   activations -- activations from feedforward pass
   L -- number of layers of neural network
   
   Returns:
   cache -- caching weigths dW, biases db for updating of parameters
   '''
   m = Y.shape[1]
   cache = {}
   cache['dA' + str(L)] = -np.divide(Y, AL) + np.divide((1-Y), (1-AL))
   cache['dZ' + str(L)] = sigmoid_backward(cache['dA' + str(L)], activations['Z' + str(L)])
   cache['dW' + str(L)] = (1/m) * np.dot(cache['dZ' + str(L)], activations['A' + str(L-1)].T)   
   cache['db' + str(L)] = (1/m) * np.sum(cache['dZ' + str(L)], axis=1, keepdims=True)
   cache['dA' + str(L-1)] = np.dot(parameters['W' + str(L)].T, cache['dZ' + str(L)])
   
   for l in range(L-1, 0, -1):
      cache['dZ' + str(l)] = relu_backward(cache['dA' + str(l)], activations['Z' + str(l)])
      cache['dW' + str(l)] = (1/m) * np.dot(cache['dZ' + str(l)], activations['A' + str(l-1)].T)   
      cache['db' + str(l)] = (1/m) * np.sum(cache['dZ' + str(l)], axis=1, keepdims=True)
      cache['dA' + str(l-1)] = np.dot(parameters['W' + str(l)].T, cache['dZ' + str(l)])
   return cache 

def update_parameters(learning_rate, parameters, cache, L):
   '''
   Update parameters with values got from backward propogation
   
   Arguments:
   learning_rate -- alpha. update with the multiple of learning rate
   parameters -- parameters for neural network (weights and biases)
   cache -- weigths dW, biases db from backward propogation
   L -- number of layers of neural network
   
   Returns:
   parameters -- updated parameters for neural network (weights and biases)
   '''
   for l in range(1, L+1):
      parameters['W' + str(l)] -= (learning_rate * cache['dW' + str(l)])
      parameters['b' + str(l)] -= (learning_rate * cache['db' + str(l)])
   return parameters
   
def n_layer_neural_network(x, y, network_dimensions, learning_rate):
   L = len(network_dimensions)
   parameters = initialize_parameters(x, network_dimensions)
   for i in range(1500):
      AL, activations = forward_propogation(x, parameters, L)
      J = cost_function(AL, y)
      #print (J)
      cache = backward_propogation(y, AL, parameters, activations, L)
      parameters = update_parameters(learning_rate, parameters, cache, L)
   print ("Cost: " + str(J))
   return parameters
   
def predict_y(predicted_output):
	return (predicted_output <= 0.5)

#Learn the network by providing inputs and outputs of XOR gate
x = np.array([[1,1],[0,0],[0,1],[1,0]]).T #shape (2, 4)
y = np.reshape([0,0,1,1], (1,4)) #shape (1, 4)
network_dimensions = [6, 4, 2, 1]
learning_rate = 0.01
parameters = n_layer_neural_network(x, y, network_dimensions, learning_rate)
#predict output
predicted_output, _ = forward_propogation(x, parameters, L = len(network_dimensions))
nn_y = predict_y(predicted_output)
for i in range(x.shape[1]):
	print (str(x[0][i]) + " " + str(x[1][i]) + " : " + str(nn_y[0][i]))

'''
parameters = initialize_parameters(x, network_dimensions)
AL, activations = forward_propogation(x, parameters, L)
J = cost_function(AL, y)
cache = backward_propogation(y, AL, parameters, activations, L)
parameters = update_parameters(learning_rate, parameters, cache, L)

#print shape of the initialized parameters
for key in parameters:
   print (key + " : " + str(parameters[key].shape))

#print shape of the activation parameters   
for key in cache:
   print (key + " : " + str(cache[key].shape))

#testing for ReLU activation
print (activations['Z1'])
print (np.maximum(np.zeros(activations['Z1'].shape), activations['Z1']))

#cost
print (J)
'''