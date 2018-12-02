# -*- coding: utf-8 -*-
"""
Created on

@author: fame
"""

import numpy as np
import matplotlib.pyplot as plt


def predict(X,W,b):
    """
    implement the function h(x, W, b) here
    X: N-by-D array of training data
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array
    """
    return sigmoid(np.dot(X, W) + b)


def sigmoid(a):
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array
    """
    return 1 / (1 + np.exp(-a))


def l2loss(X,y,W,b):
    """
    implement the L2 loss function
    X: N-by-D array of training data
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
     """

    # calculate loss
    preds = predict(X, W, b)
    loss = np.sum((y - preds)**2)

    # calculate the gradient wrt. W
    s = -2 * (y - preds) * preds * (1 - preds)
    grad_W = np.sum(X * s[:, np.newaxis], axis=0)

    # calculate the gradient wrt. b
    grad_b = np.sum(s)

    return loss, grad_W, grad_b


def train(X,y,W,b, num_iters=1000, eta=0.001):
    """
    implement the gradient descent here
    X: N-by-D array of training data
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b
     """
    losses = []
    for iter in range(num_iters):
        loss, grad_W, grad_b = l2loss(X, y, W, b)
        print("Iteration {0}, loss={1}".format(iter, loss))
        W -= eta * grad_W
        b -= eta * grad_b

        losses.append(loss)

    # plot losses
    plt.plot(list(range(1, num_iters+1)), losses)
    plt.show()

    return (W, b)

