---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Single Layer Neural Network using Python


### Implement single layer neural network using python numpy library
The training for the network is done to learn the XOR gate output
A single hidden layer network with training on 2 weight matrices and no bias vector
Activation function is sigmoid to predict output as 1 or 0


#### Import numpy library

```python
import numpy as np
```

#### Initialze random weights according to dimension provided

```python
def initialize_random(m,n):
   '''
   Initialze random weights according to dimension provided
   
   Arguments:
   m -- row dimension
   n -- column dimension
   
   Returns:
   randomly initialized matrix of dimension m*n
   '''
   return np.random.rand(m, n)
```

#### The weights by features calculation function

```python
def weights_by_features(W, x):
   '''
   The weights by features calculation function
   
   Arguments:
   x -- input vector of dimension 2*1
   W -- weight matrix
   
   Returns:
   (W.T * x) -- vector W + vector x
   '''
   return np.dot(W.T, x)
```

#### The sigmoid activation function

```python
def sigmoid(z):
   '''
   The sigmoid activation function
   
   Arguments:
   z -- vector of dimension m*n
   
   Returns:
   sigmoid of individual values in vector z in same dimension
   '''
   return (1/(1 + np.power(np.exp(z),(-1))))
```

#### Predict output value
Sigmoid value >= 0.5 is 1 else 0

```python
def predict(y):
   '''
   Sigmoid value >= 0.5 is 1 else 0
   
   Arguments:
   y -- value of sigmoid activation function
   
   Returns:
   Truth value 0 or 1
   '''
   return (1 if y > 0.5 else 0)
```

#### Difference between predicted value and actual value

```python
def end_loss(y, pred_y):
   '''
   Difference between predicted value and actual value
   
   Arguments:
   y -- actual value
   pred_y -- predicted value
   
   Returns:
   difference between y and pred_y
   '''
   return np.subtract(y, pred_y)
```

#### Calculate the error value (delta) of a layer

```python
def calculate_delta(W, next_delta, z):
   '''
   Calculate the error value (delta) of a layer
   
   Arguments:
   W -- weight matrix of to be updated of current layer
   next_delta -- value of next layer delta
   z -- value of weightmatrix to activation function of current layer
   
   Returns:
   delta -- error value of current layer
   '''
   delta = np.multiply(np.dot(W, next_delta), np.multiply(z, np.subtract(1,z)))
   return delta
```

#### Update weights according to delta of next layer and activation function of current layer

```python
def backpropogate(W, delta, a):
   '''
   Update weights according to delta of next layer and activation function of current layer
   
   Arguments:
   W -- weight matrix
   delta -- error of next layer
   a -- activation of current layer
   
   Returns:
   Updated weight matrix
   '''
   return np.add(W, np.dot(a, delta.T))
```

#### Function to implement the run of the feedforward and backpropogation of a single layer NN to learn outputs
The shapes of outputs are commented out. Can be uncommented for further clarity of network struction

```python
def networkRun(x, y, W1, W2):
   '''
   Run the feedforward and backpropogation of a single layer NN to learn outputs
   
   Arguments:
   x -- input vector with m examples of dimension n in matrix form m*n
   y -- corresponding output vector of order m*1
   W1 -- randomly initialized weight matrix of order n * ?
   W2 -- randomly initialized weight matrix of oredr ? * 1
   '''
   
   #number of iterations
   for count in range(2000):
      #print("Iteration: " + str(count))
      input = np.random.choice(np.arange(len(x))) #randomly choose input for training
      activation_layer1 = np.asmatrix(x[input]).T
      z2 = np.asmatrix(weights_by_features(W1, x[input])).T #shape 4*1
      #print ("z2: " + str(z2.shape))
      activation_layer2 = sigmoid(z2) #shape 4*1
      #print ("activation_layer2: " + str(activation_layer2.shape))
      z3 = weights_by_features(W2, activation_layer2) #shape 1*1
      #print ("z3: " + str(z3.shape))
      activation_layer3 = sigmoid(z3) #shape 1*1
      #print ("activation_layer3: " + str(activation_layer3.shape))
      pred_y = predict(activation_layer3) #output shape 1*1
      #print ("pred_y: " + str(np.asmatrix(pred_y).shape))
      print ("X: " + str(x[input]) + " Y: " + str(pred_y) + " Actual Y: " + str(y[input]))
      #run backpropogation to update weights
      delta_3 = end_loss(y[input], pred_y)
      delta_2 = calculate_delta(W2, delta_3, activation_layer2) #shape 4*1
      #print ("delta_2: " + str(delta_2.shape))
      W2 = backpropogate(W2, delta_3, activation_layer2) #shape 4*1
      #print ("W2: " + str(W2.shape))
      delta_1 = calculate_delta(W1, delta_2, activation_layer1) #shape 1*1
      #print ("delta_1: " + str(delta_1.shape))
      W1 = backpropogate(W1, delta_2, activation_layer1) #shape 2*4
      #print ("W1: " + str(W1.shape))
```

#### Learn the network by providing inputs and outputs of XOR gate

```python
x = np.array([[1,1],[0,0],[0,1],[1,0]])
y = np.array([0,0,1,1])
W1 = initialize_random(2, 4)
W2 = initialize_random(4, 1)
networkRun(x, y, W1, W2)
```

#### The network right now has learnt the weights of the XOR gate
But since the network is very basic, it's better to extend it to make it learn better for differnt kinds of input


#### Future work:
- Implement bias vector
- Use learning rate while training
- Add more layers
