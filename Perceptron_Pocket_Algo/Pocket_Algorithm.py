import random
import math
import matplotlib.pyplot as plt
import numpy as np
from preprocess_data import process_tic_tac_toe, process_sonar, process_occupancy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



class Pocket_Algorithm:
    def __init__(self):
        # Set lookup dict for kernels
        self.kernel_lookup = {
                                'dot': self._dot_kernel,
                                'poly': self._poly_kernel,
                                'cos': self._cos_kernel,
                                'yang': self._yang_kernel
                                }

    def train(self, x, y, eta, kernel, max_iter):
        '''
        Trains the perecptron on the training data using the passed kernel.
        @param x, list of training instances
        @param y, list of training labels for instances in x\
        @param eta, sets the learning rate of the algorithm
        @param kernel, lambda function for the kernel to be used
        @param max_iter, maximum number of iterations to run the algorithm

        @return a_pocket, a list containing the counts of how many times each
                    instance was misclassified during training that had the
                    longest run of consecutive correct classifications
        '''
        n = len(x)
        d = len(x[0])
        # Initialize the alpha parameter that counts each time an instance is
        # miscalculated
        a = [0] * n
        # Create pocketized version of alpha that stores the current best weights
        a_pocket = a
        run = best_run = i = 0
        # Set maximum number of times the algorithm must predict correctly 
        # before exiting
        max_run = n * 10
        kernel = self._find_kernel(kernel)
        while i < max_iter and run < max_run:
            j = random.randint(0, n - 1)
            pred = 0
            for k in range(n):
                pred += a[k] * y[k] * kernel(x[k], x[j])
            if (pred >= 0 and y[j] == -1) or (pred < 0 and y[j] == 1):
                if run > best_run:
                    best_run = run
                    a_pocket = a
                run = 0
                a[j] += 1
            else:
                run += 1
            if run > best_run: 
                best_run = run
                a_pocket = a

            i += 1

        return a_pocket

    def predict(self, x_train, y_train, x_test, y_test, a, kernel):
        '''
        Returns the predictions on the test set given the training set, the 
        alpha vector representing the number of times each data point in the
        training set was incorrectly predicted, and the passed kernel function
        '''
        kernel = self._find_kernel(kernel)
        predictions = []
        for i in range(len(x_test)):
            pred = 0
            for k in range(len(x_train)):
                pred += a[k] * y_train[k] * kernel(x_train[k], x_test[i])
            if pred >= 0:
                predictions.append(1)
            else:
                predictions.append(-1)
        return predictions

    def acc_score(self, y_test, predictions):
        return accuracy_score(y_test, predictions, normalize=True)

    def _find_kernel(self, kernel):
        if kernel in self.kernel_lookup:
            return self.kernel_lookup[kernel]

        print('This kernel is not supported, using the dot product instead.')
        return self.kernel_lookup['dot']

    def _dot_kernel(self, u, v):
        '''
        Returns the result of the dot product between vectors u and v
        '''
        prod = 0
        for i in range(len(u)):
            prod += u[i] * v[i]

        return prod

    def _poly_kernel(self, u, v):
        '''
        Returns the result of a kernel polynomial of degree 2 for vectors u and v
        '''
        pred = 0
        for i in range(len(u)):
            pred += u[i] * v[i]

        return (pred + 1)**2

    def _cos_kernel(self, u, v):
        '''
        Returns the cosine similarity between two vectors u and v
        '''
        num = 0
        for i in range(len(u)):
            num += u[i] * v[i]
        den = self._get_norm(u) * self._get_norm(v)

        return num / den

    def _yang_kernel(self, u, v):
        '''
        Returns the result of Yang Putative kernel with degree 2 for vectors u and v
        '''
        u_gt = 0
        v_gt = 0
        den = 0

        for j in range(len(u)):
            if u[j] >= v[j]:
                u_gt += u[j] - v[j]
            else:
                v_gt += v[j] - u[j]

            den += max(abs(u[j]), abs(v[j]), abs(u[j] - v[j]))
        num = ((u_gt**2) + (v_gt**2))**(1/2)

        return 1 - (num / den)

    def _get_norm(self, x):
        '''
        Returns the norm of a vector x
        '''
        out = 0
        for d in x:
            out += d**2
        return out**(1/2)

if __name__ == '__main__':
    pocket = Pocket_Algorithm()
    x, y = process_tic_tac_toe()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, 
                                        random_state=42)

    kernel = 'cos'
    a_pocket = pocket.train(x_train, y_train, 1, kernel, len(x) * 10)
    pred = pocket.predict(x_train, y_train, x_test, y_test, a_pocket, kernel)

    print(f'Tic-tac-toe accuracy, {kernel} kernel: {pocket.acc_score(y_test, pred)}')