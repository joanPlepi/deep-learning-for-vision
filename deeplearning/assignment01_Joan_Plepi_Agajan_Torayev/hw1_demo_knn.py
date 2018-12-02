# -*- coding: utf-8 -*-
"""
Created on 

@author: fame
"""

import numpy as np
from load_mnist import *
import hw1_knn  as mlBasics
from hw1_main_knn import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def main():
    X_train, y_train = load_mnist('training', [0, 1])
    X_test, y_test = load_mnist('testing', [0, 1])

    # Load data - ALL CLASSES
    # X_train, y_train = load_mnist('training')
    # X_test, y_test = load_mnist('testing')

    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    X_sampled_train, y_sampled_train = sample_the_set(X_train, y_train, 100, 2)

    """
    Flag zero will run the exercise b)
    Construct a training set with 1000 examples by randomly selecting from
    the training set, 100 samples per digit class.  Report your accuracy for
    k = 1 and k = 5 neighbors.  Visualize 1 and 5 nearest neighbor images for the first 10 test samples. 
    The confusion matrix will be plotted also.
    
    Flag 1 will run exercise c and d.
     - 5-fold cross validation in k in [1 ... 15] 
     - accuracies will be scatter plotted
     - the best k will be returned
     - knn algorithm will run in full dataset and report the accuracies for k equal to 1 and best k 
    """
    flag = 0
    if flag == 0:
        knn(X_sampled_train, y_sampled_train, X_test, y_test,1, True)
        knn(X_sampled_train, y_sampled_train, X_test, y_test,5, True)
    elif flag == 1: # exercise c and d
        best_k = cross_validation(X_sampled_train, y_sampled_train)
        print("Best k is equal to: ",)
        print("Running KNN with k equal to 1 and k equal to best k {}".format(best_k))
        knn(X_train, y_train, X_test, y_test, 1)
        knn(X_train, y_train, X_test, y_test, best_k)


def knn(X_train, y_train, X_test, y_test, k=1, plotNN=False):
    # Test on test data
    # 1) Compute distances:
    print("Calculating distances...")
    dists = mlBasics.compute_euclidean_distances(X_train, X_test)
    # 2) Run the code below and predict labels:
    print("Predicting...")
    y_pred, idx_to_plot = mlBasics.predict_labels(dists, y_train, k)
    # 3) Report results
    # you should get following message '99.91 of test examples classified correctly.'
    print('{0:0.02f}'.format(np.mean(y_pred == y_test) * 100), " of test examples classified correctly.")
    print(confusion_matrix(y_test, y_pred))

    # TODO: Can refactor this one a bit... But for now it does the work.
    if plotNN and k==1:
        fig = plt.figure(figsize=(14,8))
        for i in range(len(idx_to_plot)):
            index = idx_to_plot[i]
            pixels = X_train[index].reshape((28, 28))
            fig.add_subplot(2, 5, i+1)
            plt.imshow(pixels, cmap='gray')
        plt.show()
    elif plotNN and k >= 1:
        fig = plt.figure(figsize=(14,8))
        idx_to_plot = np.squeeze(idx_to_plot.reshape((k*10, -1)))
        for i in range(len(idx_to_plot)):
            index = idx_to_plot[i]
            pixels = X_train[index].reshape((28, 28))
            fig.add_subplot(10, 5, i+1)
            plt.imshow(pixels, cmap='gray')
        plt.show()

if __name__=="__main__":
    # Load data - two class
    main()


