# simple implementation of K-NN algorithm
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from neighbor import Neighbor, NearestNeighbors
from sklearn.metrics import accuracy_score

np.random.seed(42)

TEST_SIZE = 0.2
K = 5


X,y = load_breast_cancer(return_X_y=True)
print(f"Data Shape: {X.shape}\nLabels shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE)
print(f'Train size: {len(X_train)}\nTest size: {len(X_test)}')

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum([(xi - xj)**2 for xi,xj in zip(x1,x2)]))

def check_neighbor(test_instance, train_instance, idx):
    distance = euclidean_distance(test_instance, train_instance)
    # check if value is higher than all
    if all(i <= distance for i in nearest_neighbors.distances):
        pass
    else:
        nearest_neighbors.add_neighbor(Neighbor(train_instance, distance, y_train[idx]))
        

predictions = []

for i, test_instance in enumerate(X_test):
    nearest_neighbors = NearestNeighbors(K)

    for i, train_instance in enumerate(X_train):
        check_neighbor(test_instance,train_instance, i)
    predictions.append(nearest_neighbors.classify())

print(f'Final Accuracy: {round(accuracy_score(y_test, predictions),2)}')