import preprocess_data
import random
import math

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class Kmeans:
    def __init__(self):
        pass

    def train(self, x, y, k=2, init_centroids=None, max_iter=500, num_epochs=100):
        '''Trains kmeans clustering for the specified number of epochs, choosing
        the best result based on Sum of squared errors'''
        best_metric = float('inf')
        best_labels = None

        for i in range(num_epochs):
            labels = self._kmeans_train(x, k, init_centroids, max_iter)
            m = self._SSE(y, labels)
            if m < best_metric:
                best_metric = m
                best_labels = labels

        return best_labels      

    def _kmeans_train(self, data, k=2, init_centroids=None, max_iter=500):
        '''Runs a single iteration of the kmeans algorithm'''
        if not init_centroids:
            # No initial centroids given - choose randomly from data
            centroids = []
            for i in range(k):
                c = random.randint(0, len(data) - 1)
                while c in centroids:
                    # Ensure selected centroids have not been chosen already
                    c = random.randint(0, len(data) - 1)
                centroids.append(data[c])
        else:
            centroids = init_centroids

        # Initialize list inidicating which centroid each data point belongs to
        c_data = [-1] * len(data)
        # Initialize variable that indicates that the last iteration changed
        # centroid assignments
        changed = True
        
        it = 0
        while it < max_iter and changed:
            for i in range(len(data)):
                # initialize distance list for each centroid
                dis = []
                for j in range(len(centroids)):
                    dis.append(self._distance(data[i], centroids[j]))
                closest = dis.index(min(dis))
                changed = changed and (closest == c_data[i])
                c_data[i] = closest
            centroids = self._update_centroids(data, c_data, centroids)
            it += 1
        return c_data

    def _update_centroids(self, data, c_data, centroids):
        '''Updates the centroids based on the current cluster assignments'''
        d = [0] * len(centroids[0])
        new_centroids = [d] * len(centroids)
        totals = [0] * len(centroids)

        for i in range(len(data)):
            new_centroids[c_data[i]] = self._vector_sum(new_centroids[c_data[i]], data[i])
            totals[c_data[i]] += 1

        for i in range(len(new_centroids)):
            if totals[i] == 0:
                # Avoid divide by zero if some cluster has no points assigned to it
                # Only possible if init_centroids were not data points
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = [x / totals[i] for x in new_centroids[i]]

        return new_centroids

    def _distance(self, x, y):
        '''Finds the distance between the points x and y'''
        dis = 0
        for i in range(len(x)):
            dis += (x[i] - y[i])**2
        return math.sqrt(dis)

    def _vector_sum(self, x, y):
        '''Sums two vectors of the same size together'''
        s = [0] * len(x)
        for i in range(len(x)):
            s[i] += x[i] + y[i]
            
        return s

    def _SSE(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)



def plot_data(x, y, preds):
    colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'yellow', 4: 'black' }
    ticks = {0: '^', 1: 'o', 2: '+'}
    for i in set(y):
        cur_data = []
        color = []
        for j in range(len(x)):
            if y[j] == i:
                cur_data.append([x[j][0], x[j][1]])
                color.append(colors[preds[j]])
        tick = ticks[i]
        x1 = [i[0] for i in cur_data]
        x2 = [i[1] for i in cur_data]

        plt.scatter(x1, x2, c=color, marker=tick)
    plt.show()

if __name__ == '__main__':
    x, y = preprocess_data.load_data('data/clustering_dataset_1.txt')

    kmeans = Kmeans()
    clusters = kmeans.train(x, y, k=3)

    plot_data(x, y, clusters)

