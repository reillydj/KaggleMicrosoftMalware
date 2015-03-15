__author__ = 'David Reilly'

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

class logistic_regression():

    def __init__(self, regularization=None, gradient_descent=None, threshold=0.5):

        self.regularization = regularization
        self.gradient_descent = gradient_descent
        self.threshold = threshold
        self.type = "LogisticRegression"
        self.coefficients = np.zeros(1)
        self.p_values = None
        self.iterations_to_convergence = 0

    def fit(self, train_data, labels):

        assert not np.any(np.isnan(train_data)), "Dataset contains Null Values"

        self.coefficients = np.random.rand(train_data.shape[1] + 1)
        train_data = np.c_[np.ones(len(train_data)), train_data]
        self.stochastic_gradient_descent(train_data, labels, 0.005, 1.0)


    def predict(self, test_data, threshold=0.5):

        probabilities = self.logit_link_function(np.dot(np.c_[np.ones(len(test_data)), test_data], self.coefficients))
        probabilities[probabilities > threshold] = 1
        probabilities[probabilities < threshold] = 0
        return list(probabilities)

    def predict_proba(self, test_data):

        return self.logit_link_function(np.dot(np.c_[np.ones(len(test_data)), test_data], self.coefficients))

    def logit_link_function(self, x):

        return 1.0 / (1.0 + np.exp(-x))

    def stochastic_gradient_descent(self, train_data, labels, learning_rate, c):

        precision = np.empty(train_data.shape[1])
        precision.fill(0.000001)
        while True:

            next_coefficients = np.copy(self.coefficients)
            for i, label in enumerate(labels):
                next_coefficients += learning_rate * self.stochastic_update(train_data[i, :], label, next_coefficients)\
                                     - learning_rate * 1.0 / c * next_coefficients
                self.iterations_to_convergence += 1
                if all(abs(next_coefficients - self.coefficients) < precision):
                    break

            if all(abs(next_coefficients - self.coefficients) < precision):
                break
            self.coefficients = next_coefficients

    def stochastic_update(self, row, label, coefficients):
        return (label - self.logit_link_function(np.dot(coefficients, row))) * row

if __name__ == '__main__':

    data = load_iris()
    # print data.target
    y = data.target[[0, 1, 2, 25, 45, 47, 50, 54, 55, 58]]
    X = data.data[[0, 1, 2, 25, 45, 47, 50, 54, 55, 58], :]
    X_test = data.data[[3, 27, 46, 51, 53, 56], :]
    print "TRUE Y = ", data.target[[3, 27, 46, 51, 53, 56]]
    # print X

    test = logistic_regression()
    sklearn = LogisticRegression(penalty="l2")
    sklearn.fit(X, y)
    print "SKLEARN PREDICTIONS = ", sklearn.predict(X_test)
    print "SKLEARN PROBABILITIES = ", sklearn.predict_proba(X_test)
    print "SKLEARN COEFFS = ", sklearn.coef_
    test.fit(X, y)
    print "MY PREDICTIONS = ", test.predict(X_test)
    print "MY PROBABILITIES = ", test.predict_proba(X_test)
    print "MY COEFFS = ", test.coefficients
    print "CONVERGED IN ", test.iterations_to_convergence, " ITERATIONS"
    # test.stochastic_gradient_descent(np.array([[1, 2, 3], [3, 4, 5]]), np.array([1, 0]), 0.00001)