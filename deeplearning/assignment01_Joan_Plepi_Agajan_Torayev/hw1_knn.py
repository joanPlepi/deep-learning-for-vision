# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np
from time import time

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """
    tic = time()
    dist = -2*np.dot(Y, X.T) + np.sum(Y**2, axis=1).reshape(-1,1) + np.sum(X**2, axis=1)
    print("Elapsed {:f} sec calculating the distances".format(time() - tic))
    return dist


def predict_labels(dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    tic = time()
    if k == 1:
        idx_of_knn = np.argmin(dists, axis=1)
        idx_to_return = idx_of_knn[:10]
        pred_labels = labels[idx_of_knn]

    else:
        idx = np.argpartition(dists, range(k), axis=1)
        idx_of_knn = idx[:, :k]
        idx_to_return = idx_of_knn[:10, :k]

        from scipy.stats import mode
        pred_labels = mode(labels[idx_of_knn], axis=1)[0].reshape(-1)

    # print("Time to calculate NN for k equal to {} was {:f} sec".format(k, time() - tic))
    return pred_labels, idx_to_return


# def get_weighted_sums(dists, i, idx, labels):
#     temp = {}
#     for j in idx:
#         if (labels[j] not in temp.keys()):
#             temp[labels[j]] = dists[j, i]
#         else:
#             temp[labels[j]] += dists[j, i]
#     return temp