import random
import time

import numpy as np


class NeuralNetwork(object):

    def __init__(self, layers_sizes):
        self.layers = len(layers_sizes)
        self.biases = [np.random.randn(y, 1) for y in layers_sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers_sizes[:-1], layers_sizes[1:])]


    def fit(self, training_data: zip, validation_data: zip, epochs: int=20, batch_size: int=10, c=3.0, verbose=False):
        training_data = list(training_data)
        validation_data = list(validation_data)

        n = len(training_data)
        n_val = len(validation_data)

        for epoch in range(epochs):
            start = time.time()
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.batch_fit(batch, c)
            end = time.time()
            if verbose:
                #accuracy_tr = self.accuracy(training_data, n)
                accuracy_val = self.accuracy(validation_data, n_val)
                print(f"Epoch {epoch}:: Validation Accuracy: {accuracy_val}, Time: {end-start}")


    def predict(self, test_data: zip):
        test_data = list(test_data)
        preds = [np.argmax(self.feedforward(x)) for x, _ in test_data]
        return(preds)


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a


    def accuracy(self, validation_data: zip, n_val: int):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in validation_data]
        return sum(int(x == y) for (x, y) in test_results) / n_val


    def batch_fit(self, batch, c):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in batch:
            delta_grad_b, delta_grad_w = self.backpropagation(x, y)
            grad_b = [nb+dnb for nb, dnb in zip(grad_b, delta_grad_b)]
            grad_w = [nw+dnw for nw, dnw in zip(grad_w, delta_grad_w)]
        self.weights = [w-(c/len(batch))*nw
                        for w, nw in zip(self.weights, grad_w)]
        self.biases = [b-(c/len(batch))*nb
                       for b, nb in zip(self.biases, grad_b)]

    def backpropagation(self, x, y):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        activation = x
        activations = [x] # list with all activations
        zs = [] # list to store all the z vectors for the back process
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # backpropagation
        delta = self.d_cost(activations[-1], y) * self.d_sigmoid(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.layers):
            z = zs[-l]
            ds = self.d_sigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * ds
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (grad_b, grad_w)


    @staticmethod
    def d_cost(output_activations, y):
        return 2*(output_activations-y)


    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))


    def d_sigmoid(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
