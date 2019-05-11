#!/usr/bin/env python
# coding: utf-8

# ## Neural network (with one hidden layer) from scratch ##
# #### Purpose of the neural network : 2 class classification ####
# ### Learning objective: various activation functions, compute cross entropy loss, forward & backward propagation, random initialization ###

# ### Input Data ###

# Here input data X : has two input features (both are real numbers)
# Y : has two classes 0, 1
# We'll generate X, Y with help of code

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

# function to create data set
# create a flower like structure

def load_planar_dataset(seed=1):
    np.random.seed(seed)
    m = 400 # number of examples
    N = int(m/2) # number of points per class; N=200
    D = 2 # dimensionality; two classes
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))# 0-199; 200-399;
        
        # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
        # Return evenly spaced numbers over a specified interval
        
        # numpy.random.randn(d0, d1, ..., dn)
        # d0, d1, …, dn : int, optional - The dimensions of the returned array
        # Return a sample (or samples) from the “standard normal” distribution
        
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)] 
        # numpy.c_ Translates slice objects to concatenation along the second axis
        
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y



X, Y = load_planar_dataset()


# ### Simple Logistic Regression from Scikit learn - just to compare with NN model to be developed later ###

classifier = sklearn.linear_model.LogisticRegressionCV(); # logisic regression with built is cross validation
classifier.fit(X.T, Y.T); 

# Print accuracy
LR_predictions = classifier.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")



# ### Neural Network Model ###

# we will have the following model

# Input layer - [0] : X = A0 - two input features - so n0=2; A0 - [n0 x m]
# One hidden layer - [1] - n1 hidden units - n1=4 ; Activation function: tanh
# Output layer [2] - n2 unit - n2=1; Activation function: sigmoid

# Parameters for [1]
# W1 - [n1 x n0] = [4 x 2] matrix
# b1 - [n1 x 1] = [4 x 1] matrix

# Parameters for [2]
# W2 - [n2 x n1] = [1 x 4] matrix
# b2 - [n2 x 1] = [1 x 1] matrix

# Forward Pass
# Z1 = W1 x A0 + b1 => [n1 x n0] x [n0 x m] + [n1 x 1] ==> [n1 x m]
# A1 = tanh(Z1) => [n1 x m]
# Z2 = W2 x A1 + b2 => [n2 x n1] x [n1 x m] + [n2 x 1] ==> [n2 x m]
# Yhat= A2 = sigmoid(Z2) => [n2 x m] = [1 x m] -- this is same as dimension of Y

# Cost function
# This is binary classification - so cost function is as follows:
# J = (-1/m)Sum (Y*log(Yhat)+(1-Y)*log(1-Yhat))

# Backward Pass
# dZ2 = A2 - Y; dimension [n2 x m]
# dW2 = (1/m) dZ2 x A1.T ; dim check - [n2 x m] x [m x n1] => [n2 x n1] -- same as W2
# db2 = (1/m) Sum(dZ2, column-wise sum) => [n2 x 1] -- same as b2
# dZ1 = W2.T x dZ2 * g1prime(Z1) => [n1 x n2] x [n2 x m] * [n1 x m] => [n1 x m]; g1prime - derivative of g1 i,e, tanh
# dW1 = (1/m) dZ1 x A0.T; [n1 x m] x [m x n0] => [n1 x n0] -- same as W1
# db1 = (1/m) Sum(dZ1, column-wise sum) => [n1 x 1] -- same as b1

# Update
# W1 = W1 - alpha.dW1 and so on

# What are variables
# n0, n1, n2; 

# Functions to write in order to modularize
# Initialize parameters - arg: n0, n1, n2; function: random initialization of W,b s and return them
# Forward Pass function - arg: A0=X, W, b; function: calculate A2 and J
# Backward Pass function - arg: A2, Y; function: calculate derivatives and update W,b
# Predict function - arg: X_test; function: calculate Yhat_test and return

def initialize_parameters(numberOfLayers, listNumberOfUnits):
    """
    numberOfLayers - int - should include input layer also
    listNumberOfUnits - array of int - units in each layer
    
    Returns:
    a dictionary - parameters for W's and b's
    """
    parameter = {} # start with empty dictionary
    
    for layer in range(1,numberOfLayers):
        print(listNumberOfUnits[layer],listNumberOfUnits[layer-1])
        Wlayer = np.random.randn(listNumberOfUnits[layer], listNumberOfUnits[layer-1]) * 0.01
        blayer = np.zeros((listNumberOfUnits[layer],1))
        parameter["W"+str(layer)] = Wlayer
        parameter["b"+str(layer)] = blayer
    
    return parameter


def forward_pass(X,Y, parameter):
    """
    X - input - dimension [n0 x m]; Note: X=A0; n0=nx;
    Y - ground truth - [nL x m]
    parameter - dictionary - contains W's, b's
    
    Returns:
    activation values - dictionary
    J - cost 
    """
    
    m = X.shape[1]
    numberLayers = int(len(parameter)/2) # this is without input layer - if len(parameter)==10 - then means 5 layers
    A_prev = X
    J = 0
    activation = {} # start with empty dictionary
    activation['0'] = X
    
    for layer in range(1,numberLayers+1): # i.e. 1 to 5
        W = parameter["W" + str(layer)]
        b = parameter["b" + str(layer)]
        #print('layer' + str(layer))
        #print('shape: W')
        #print(W.shape)
        #print('shape: A_prev')
        #print(A_prev.shape)
        Z = np.dot(W,A_prev) + b
        if layer != numberLayers:
            A = activationFunction(Z, 'tanh')
        else:
            A = activationFunction(Z, 'sigmoid')
        activation[str(layer)]= A
        A_prev = A
    
    # at end of iteration, Yhat = A
    
    J = (-1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))
    
    return activation,J


# Write the activation function 
# Assumption: hidden layer - tanh activation function; output layer - sigmoid function
def activationFunction(Z, fn='tanh'):
    """
    Z - a matrix
    layer - int - indicating the current layer
    
    Returns:
    result of activation 
    """
    if fn== 'sigmoid':
        # sigmoid
        A = 1/(1+np.exp(-Z))
    else:
        # tanh
        A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    
    return A


# Backward Pass
# dZ2 = A2 - Y; dimension [n2 x m]
# dW2 = (1/m) dZ2 x A1.T ; dim check - [n2 x m] x [m x n1] => [n2 x n1] -- same as W2
# db2 = (1/m) Sum(dZ2, column-wise sum) => [n2 x 1] -- same as b2
# dZ1 = W2.T x dZ2 * g1prime(Z1) => [n1 x n2] x [n2 x m] * [n1 x m] => [n1 x m]; g1prime - derivative of g1 i,e, tanh
### derivative of tanh(Z) = 1- tanh(Z)^2
# dW1 = (1/m) dZ1 x A0.T; [n1 x m] x [m x n0] => [n1 x n0] -- same as W1
# db1 = (1/m) Sum(dZ1, column-wise sum) => [n1 x 1] -- same as b1

# Backward Pass function - arg: A2, Y; function: calculate derivatives and update W,b

def backward_pass(Yhat, Y, parameter, activation, alpha=0.01):
    """"
    Yhat, Y - [nL x m]
    parameter - dictionary containing W's, b's
    activation - dictionary conatining activation values
    
    Returns:
    updated parameters
    """
    
    m = Y.shape[1]
    numberLayers = int(len(parameter)/2) # this is without the input layer
    
    ## shall try the loop based implementation layer
    ## now lets do basic calculation
    dZ2 = Yhat - Y
    dW2 = (1/m)*np.dot(dZ2, activation['1'].T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(parameter["W2"].T,dZ2) * (1 - np.power(activation['1'], 2))# g1prime -- tanh-derivative - 1-a^2;
    dW1 = (1/m)* np.dot(dZ1, activation['0'].T)
    db1 = (1/m)* np.sum(dZ1, axis=1, keepdims=True)
    
    parameter['W1'] = parameter['W1'] - alpha* dW1
    parameter['b1'] = parameter['b1'] - alpha* db1
    parameter['W2'] = parameter['W2'] - alpha* dW2
    parameter['b2'] = parameter['b2'] - alpha* db2
    
    return parameter


## actual running the model
np.random.seed(3)
numberOfIteration = 20000
numberOfLayers = 3
listNumberOfUnits = [2,4,1]
alpha = 1.2
parameter = initialize_parameters(numberOfLayers, listNumberOfUnits)
for i in range(numberOfIteration):
    #print('i '+str(i))
    activation,J = forward_pass(X,Y, parameter)
    if (i%1000==0):
        print('Iteration '+str(i))
        print('Cost' + str(J))
    
    Yhat = activation[str(numberOfLayers-1)]
    parameter = backward_pass(Yhat, Y, parameter, activation, alpha)
    
print (parameter['W1'])

print(parameter['b1'])

print(parameter['W2'])

parameter['b2']


# Predict function - arg: X_test; function: calculate Yhat_test and return
def predict_class(X_test, parameters):
    """
    X - input - dimension [n0 x m]; Note: X=A0; n0=nx;
    parameter - dictionary - contains W's, b's
    
    Returns:
    Yhat - [nL x m] matrix
    """
    
    m = X_test.shape[1]
    numberLayers = int(len(parameter)/2) # this is without input layer - if len(parameter)==10 - then means 5 layers
    A_prev = X_test
    
    for layer in range(1,numberLayers+1): # i.e. 1 to 5
        W = parameter["W" + str(layer)]
        b = parameter["b" + str(layer)]
        Z = np.dot(W,A_prev) + b
        if layer != numberLayers:
            A = activationFunction(Z, 'tanh')
        else:
            A = activationFunction(Z, 'sigmoid')
        A_prev = A
    
    # at end of iteration, Yhat = A
    # Now if probability > 0.5 - we'll classify as 1; otherwise 0
    Yhat = A>0.5
    
    return Yhat    


Predictions = predict_class(X, parameter)

print ('Accuracy: %d' % float((np.dot(Y,Predictions.T) + np.dot(1-Y,1-Predictions.T))/float(Y.size)*100) + '%')


# ### Learnings ###
# #### Generate data set ####
# #### Write initialization, forward, backward, predict functions ####
# #### Idea on a simple NN with one hidden layer written from scratch ####

# The above coding is inspired by the deep learning course of Prof. Andrew Ng
