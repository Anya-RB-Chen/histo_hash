import numpy as np
import time
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import random

class LSH:
    def __init__(self, n_estimators=10, n_candidates=50, random_state=None):
        """
            n_estimators: The number of hash tables to use
            n_candidates: The number of candidates to consider for each query
            random_state: The seed used by the random number generator
        """
        self.n_estimators = n_estimators
        self.n_candidates = n_candidates
        self.random_state = check_random_state(random_state)
        

    def fit(self, X):
        self.n_features = X.shape[1]
        self.estimators = []
        for i in range(self.n_estimators):
            estimator = LSH_Estimator(self.n_features, self.n_candidates, self.random_state)
            estimator.fit(X)
            self.estimators.append(estimator)

    def kneighbors(self, X, n_neighbors=5):
        distances = np.full((X.shape[0], n_neighbors), np.inf)
        indices = np.full((X.shape[0], n_neighbors), -1, dtype=int)
        e_count = 0

        for estimator in self.estimators:
            
            candidate_indices = estimator.get_candidates(X)
            for i in range(X.shape[0]):
                candidates = X[candidate_indices[i]]
                d = np.linalg.norm(candidates - X[i], axis=1)
                for j in range(d.shape[0]):
                    if d[j] < np.max(distances[i]):
                        index = np.argmax(distances[i])
                        distances[i, index] = d[j]
                        indices[i, index] = candidate_indices[i][j]
            # print('Estimator {} done'.format(e_count))
            e_count += 1

        return distances, indices


class LSH_Estimator:
    def __init__(self, n_features, n_candidates, random_state):
        """
            n_features: The number of features in the dataset
            n_candidates: The number of candidates to consider for each query
            random_state: The seed used by the random number generator
        """
        self.n_features = n_features
        self.n_candidates = n_candidates
        self.random_state = random_state

    def fit(self, X):
        self.offsets_ = self.random_state.uniform(0, 1, size=self.n_features)
        self.X_ = X

    def get_candidates(self, X):
        bins = (X + self.offsets_) // 1
        candidate_indices = []
        for i in range(X.shape[0]):
            candidates = set()
            for j in range(self.X_.shape[0]):
                # print(bins[i] - ((self.X_[j] + self.offsets_) // 1))
                if np.all(bins[i] == (self.X_[j] + self.offsets_) // 1):
                    candidates.add(j)
                    if len(candidates) >= self.n_candidates:
                        break
            candidate_indices.append(list(candidates))
        return candidate_indices


# ###############################################################################################
# # Implementation of LSH for MNIST dataset
# ###############################################################################################

# # Load MNIST dataset
# # X.shape = (60000, 784)
# mnist = pd.read_csv('./src/dataset/mnist_train.csv')
# X = mnist.values

# # Divide dataset for algorithm and query respectively
# data_algorithm = X[:50000, 1:]
# data_query = X[50000:, 1:]

# # Normalize data
# data_algorithm = data_algorithm.astype('float64') / 255
# data_query = data_query.astype('float64') / 255

# # Create and fit LSH model
# lsh = LSH(random_state=5, n_estimators=10, n_candidates=1000)
# lsh.fit(data_algorithm)

# # Find nearest neighbors of the first image in the dataset
# start = time.time()
# lsh_distances, lsh_indices = lsh.kneighbors(data_algorithm[0].reshape(1, -1), n_neighbors=10)
# end = time.time()

# #################################################################################################
# # Evaluation of LSH for MNIST dataset
# #################################################################################################
# # Create and fit NearestNeighbors model
# nn = NearestNeighbors()
# nn.fit(data_algorithm)
# # Find nearest neighbors of the first image in the dataset using NearestNeighbors
# nn_distances, nn_indices = nn.kneighbors(data_algorithm[0].reshape(1, -1), n_neighbors=10)

# # Compute recall
# recall = len(set(nn_indices[0]).intersection(set(lsh_indices[0]))) / len(nn_indices[0])
# print(f'Recall: {recall}')

# print('Time taken LSH in MNIST: ', end - start)
# print('Indices of nearest neighbors using LSH: ', lsh_indices)
# print('Distances of nearest neighbors using LSH: ', lsh_distances)
# print('Indices of nearest neighbors using NearestNeighbors: ', nn_indices)
# print('Distances of nearest neighbors using NearestNeighbors: ', nn_distances)
