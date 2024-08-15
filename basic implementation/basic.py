# MNIST digit classification Using only numpy and math

#   28 * 28 pixel images, M number of images.
#   Each column is an image. 28 * 28 values per column

import numpy as np
import pandas as pd
data = pd.read_csv('../digits/train.csv')
data = np.array(data)
m, n = data.shape

np.random.shuffle(data)

# Splitting into training and dev data
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

print(X_train.shape)

def init_parameters():
    W1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5

    W2 = np.random.rand(64, 128) - 0.5
    b2 = np.random.rand(64, 1) - 0.5

    W3 = np.random.rand(10, 64) - 0.5
    b3 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2, W3, b3

def recLu(Z):
    return np.maximum(Z, 0)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def softMax(Z):
    return np.exp(Z)/sum(np.exp(Z))

def forward_prop(w1, b1, w2, b2, w3, b3,  X):
    # Computing forward. Input to Output
    Z1 = w1.dot(X) + b1
    A1 = recLu(Z1)

    Z2 = w2.dot(A1) + b2
    A2 = recLu(Z2)

    Z3 = w3.dot(A2) + b3
    A3 = softMax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def recLu_derivative(Z):
    return Z > 0

# def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
#     one_hot_Y = one_hot(Y)
#     dZ2 = A2 - one_hot_Y
#     dW2 = 1 / m * dZ2.dot(A1.T)
#     db2 = 1 / m * np.sum(dZ2)
#     dZ1 = W2.T.dot(dZ2) * recLu_derivative(Z1)
#     dW1 = 1 / m * dZ1.dot(X.T)
#     db1 = 1 / m * np.sum(dZ1)
#     return dW1, db1, dW2, db2

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = X.shape[1]  # Number of samples
    one_hot_Y = one_hot(Y)

    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * np.dot(dZ3, A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * recLu_derivative(Z2)
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * recLu_derivative(Z1)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    # 2 layers, input layer[0th] has 28 * 28 input neurons.
    # Hidden Layer has 10 neurons. Output layer has 10 neurons, each for a digit.
    W1, b1, W2, b2, W3, b3 = init_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y) #Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)   #W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print("Accuracy : ", round(100*get_accuracy(predictions, Y), 3))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

