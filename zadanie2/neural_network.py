import random
import time

import numpy as np


class NeuralNetwork(object):

    def __init__(self, layers_sizes: list, act_func_names="sigmoid"):
        self.layers = len(layers_sizes)
        self.biases = [np.random.randn(n_l, 1) for n_l in layers_sizes[1:]]
        self.weights = [np.random.randn(n_l, x) for x, n_l in zip(layers_sizes[:-1], layers_sizes[1:])]

        if isinstance(act_func_names, str):
            self.act_func_names = len(layers_sizes) * [act_func_names]
        else:
            self.act_func_names = act_func_names


    def fit(self, training_data: zip, validation_data: zip, epochs: int=20, batch_size: int=10, c: float=3.0, verbose: bool=False):
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
                accuracy_val = self.accuracy(validation_data, n_val)
                print(f"Epoch {epoch}:: Validation Accuracy: {accuracy_val}, Time: {end-start}")


    def predict(self, test_data: zip):
        test_data = list(test_data)
        preds = [np.argmax(self.feedforward(x)) for x, _ in test_data]
        return(preds)


    def feedforward(self, a: np.array):
        for b, w, f_name in zip(self.biases, self.weights, self.act_func_names):
            a = self.activation(f_name, np.dot(w, a)+b)
        return a


    def accuracy(self, validation_data: zip, n_val: int):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in validation_data]
        return sum(int(x == y) for (x, y) in test_results) / n_val


    def batch_fit(self, batch: np.array, c: float):
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


    def backpropagation(self, x: np.array, y: np.array):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        activation = x
        activations = [x] # list with all activations
        zs = [] # list to store all the z vectors for the back process
        for b, w, f_name in zip(self.biases, self.weights, self.act_func_names):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation(f_name, z)
            activations.append(activation)

        # backpropagation
        delta = self.d_cost(activations[-1], y) * self.d_activation(f_name, zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.layers):
            z = zs[-l]
            ds = self.d_activation(self.act_func_names[self.layers - l ], z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * ds
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (grad_b, grad_w)


    @staticmethod
    def d_cost(output_activations: np.array, y: np.array):
        return 2*(output_activations-y)


    @staticmethod
    def activation(function_name: str, x: float):
        match function_name:
            case "sigmoid":
                return 1.0/(1.0+np.exp(-x))
            case "relu":
                relu_func = np.vectorize(lambda x: max(0, x))
                return relu_func(x)
            case "tanh":
                return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


    def d_activation(self, function_name: str, x: float):
        match function_name:
            case "sigmoid":
                y = self.activation("sigmoid", x)
                return y*(1-y)
            case "relu":
                d_relu_func = np.vectorize(lambda x: 1 if x>0 else 0)
                return d_relu_func(x)
            case "tanh":
                return 1 - self.activation("tanh", x)**2
