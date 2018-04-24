import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LinearRegression():

    def __init__(self, normalize=True):
        self._normalize = normalize
        self._mean = 0
        self._std = 1
        self.theta = np.zeros(2)

    def _normalize_dataset(self, X, verbose=False):
        self._mean = X.mean()
        self._std = X.std()
        normalized = (X - self._mean) / self._std
        if verbose:
            print(X)
            print(normalized)
        return normalized

    def _normalize_examples(self, x):
        if self._normalize:
            return (x - self._mean) / self._std
        else:
            return x

    def _hypothesis(self, X, theta):
        return theta[0] + theta[1] * X

    def _cost(self, X, y, theta):
        m = len(X)
        return (1 / (2 * m)) * sum((self._hypothesis(X, theta) - y) ** 2)

    def _gradient_step(self, X, y, theta, lr, verbose=False, stepNo=None, plot=False):
        m = len(X)
        tmp0 = (lr / m) * sum(self._hypothesis(X, theta) - y)
        tmp1 = (lr / m) * sum((self._hypothesis(X, theta) - y) * X)
        if verbose:
            print("Iteration  [%d]" % (stepNo))
            print("Theta0: %f - %f = %f" %(theta[0], tmp0, theta[0] - tmp0))
            print("Theta1: %f - %f = %f" %(theta[1], tmp1, theta[1] - tmp1))
            print()
        theta[0] -= tmp0
        theta[1] -= tmp1
        if plot:
            self.display(theta)
        return theta

    def train(self, X, y, theta=None, n_iter=1500, lr=0.01, verbose=False, visual=False):
        X = np.array(X)
        y = np.array(y)
        if self._normalize:
            X = self._normalize_dataset(X)
        if theta is None:
            theta = self.theta

        # Perform gradient descent
        costs = []
        for i in range(n_iter):
            costs.append(self._cost(X, y, theta))
            theta = self._gradient_step(X, y, theta, lr, verbose=verbose, stepNo=(i+1))

        # Visualize error diminution
        if visual:
            fig = plt.figure("Cost")
            fig.suptitle("Cost function minimization")
            ax = plt.axes()
            plt.xlabel("Nb of iterations")
            plt.ylabel("Cost (MSE)")
            ax.plot(costs)
            plt.show()

        self.theta = theta
        return theta

    def predict(self, X, theta=None):
        X = np.array(X)
        if theta is None:
            theta = self.theta
        if self._normalize:
            X = self._normalize_examples(X)
        return self._hypothesis(X, theta)

    def display(self, X, y):
        fig = plt.figure("Data visualization")
        fig.suptitle("Data visualization and regression line")
        ax = plt.axes()
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price ($)")
        plt.scatter(X, y)
        reg_line = self.theta[1] * self._normalize_examples(X) + self.theta[0]
        ax.plot(X, reg_line, 'r-', X, y, 'o')
        plt.show()
