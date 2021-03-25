import numpy as np
from scipy.stats import mode

class Neighbor():

    def __init__(self, features = 0, distance = float('inf'), label = None):
        self.features = features
        self.distance = distance
        self.label = label

    def __repr__(self):
        return f"Features: {self.features}\tDistance: {self.distance}\tLabel: {self.label}"

class NearestNeighbors():

    def __init__(self, k):
        self.k = k
        self.init_neighbors()
        self.update_distances()

    def init_neighbors(self):
        self.neighbors = np.full(self.k, Neighbor())

    def update_distances(self):
        self.distances = [n.distance for n in self.neighbors]

    def add_neighbor(self, neighbor):
        self.neighbors[np.argmax(self.distances)] = neighbor
        self.update_distances()

    def classify(self):
        return mode([n.label for n in self.neighbors])[0][0]

    def __repr__(self):
        return str([n for n in self.neighbors])
        