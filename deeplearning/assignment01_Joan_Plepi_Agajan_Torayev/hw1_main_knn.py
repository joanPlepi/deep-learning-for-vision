from time import time

import numpy as np
from numpy import random
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import hw1_knn  as mlBasics
import matplotlib.pyplot as plt


def sample_the_set(X, y, sampling_size, nr_of_classes):
    tic = time()
    x_sampled = np.zeros((sampling_size * nr_of_classes, X.shape[1]))
    y_sample = np.zeros((sampling_size * nr_of_classes))

    for i in range(nr_of_classes):
        idx_per_label = np.where(y == i)
        # np.where is returning a tuple, so I do idx_per_label[0] to get the array of indices.
        sampled_indexes = random.choice(idx_per_label[0], sampling_size)
        x_sampled[sampling_size*i:sampling_size*(i+1)] = X[sampled_indexes]
        y_sample[sampling_size*i:sampling_size*(i+1)] = y[sampled_indexes]

    print("Time for sampling {:2f}".format(time()-tic))

    # TODO: Do we need to shuffle in this particular case?
    return shuffle(x_sampled, y_sample)

def cross_validation(X_train, y_train):
    results = np.zeros(15)
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    size_for_fold = int(X_train.shape[0] / n_folds)

    # We precompute the distances for efficiency here.
    dists = np.zeros((n_folds, size_for_fold, (n_folds - 1) * size_for_fold))
    # for i, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
    #     dists[i] = mlBasics.compute_euclidean_distances(X_train[train_index, :], X_train[test_index, :])

    # TODO: Check here the cross validation
    for k in range(1, 4):
        temp_results = np.zeros(n_folds)
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            dists[i] = mlBasics.compute_euclidean_distances(X_train[train_index, :], X_train[test_index, :])
            y_test_pred = mlBasics.predict_labels(dists[i], y_train[train_index], k)
            temp_results[i] = np.mean(y_test_pred == y_train[test_index]) * 100

        results[k - 1] = temp_results.mean()
        print("Accuracy for k {}: {:2f}".format(k, results[k-1]))
    plot(results)

    best_k = np.argmax(results) + 1
    return best_k

def show_image(matrix_to_show):
    img = matrix_to_show[1][0]

    print(img.shape)
    plt.imshow()

def plot(results):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.scatter([i + 1 for i, _ in enumerate(results)], results)
    axes.set_xlabel('Number of Neighbors K')
    axes.set_ylabel('Accuracy')
    axes.set_ylim(70,100)
    axes.set_xlim(1,20)
    plt.show()