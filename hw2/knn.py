import numpy as np


def knn(train_set, train_labels, test_sample, k):
    dists_array = [np.linalg.norm(item - test_sample) for item in train_set]
    dists_array = np.array(dists_array)
    k_nearest_neighbours = dists_array.argsort()[:k]
    knn_labels = [train_labels[i] for i in k_nearest_neighbours]
    label = 0 if sum(knn_labels)/len(knn_labels) < 0.5 else 1

    return label

