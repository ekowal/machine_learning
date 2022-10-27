#!/usr/bin/env python3

import numpy as np
import pytest
from sklearn.linear_model import Ridge

class RidgeRegr:
    """
    Ridge Regresion - least squares with regularization term 
    Minimizes following function of theta:

    L(theta) = (Y - X@theta)^T@(Y - X@theta) + alpha*theta^T@theta

    using gradient descent algorithm.

    Attributes
    ----------
    alpha: float, default 0, non negative constant which controls 
        regularization term
    max_iter: int, default 1e6, maximum number of iterations in gradient
        descent algorithm
    c: float, default 1e-3, step size (learning rate) in gradient descent 
        method
    eps: float, default 1e-8, if ||new_theta-theta||_1<eps gradient 
        descent stops
    theta: array of ridge regression coefficients


    """
    def __init__(self, alpha = 0.0, max_iter = 1000000, c=1e-3, eps=1e-8):
        self.alpha = alpha
        self.c = c
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, X, Y):
        # wejscie:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Znajduje theta (w przyblizeniu) minimalizujace kwadratowa funkcje kosztu L uzywajac metody iteracyjnej.
        n, m = X.shape
        X = np.hstack((np.ones((n,1)), X))
        theta = np.zeros(m+1) #np.random.normal(size=m+1)

        for _ in range(self.max_iter):
            grad = -2*X.T@Y + 2*theta@X.T@X + 2*self.alpha*np.concatenate((np.zeros(1), theta[1:]))
            new_theta = theta - self.c * grad
            if np.sum(abs(new_theta-theta)) < self.eps:
                break
            theta = new_theta

        self.theta = theta
        return self
    
    def predict(self, X):
        # wejscie
        #  X = np.array, shape = (k, m)
        # zwraca
        #  Y = wektor(f(X_1), ..., f(X_k))
        k, _ = X.shape
        ones = np.ones(k).reshape(-1,1)
        X = np.hstack((ones, X))
        return X@self.theta



def test_RidgeRegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,2,10]).reshape((3,1))
    alpha = 0.3
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)
    

def test_RidgeRegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)