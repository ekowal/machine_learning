import random
import time

import numpy as np


class NeuralNetwork():
    """
    Batch learning neural network. The weights 
    and biases are being updated through gradient
    descent using backpropagation on batches

    Atributes
    ---------
    layers_sizes: list of layers sizes
    weights_init: weights and biases initialization method, possible
        options are "random", "xavier" or "zeros"
    act_func_names: str or list containing names of activation
        functions for the respective layers, if value is string
        then the same act func will be used for all layers
    weights: list containing weights for all respective layers
    biases: list containing biases for all respective layers


    """
    def __init__(self, layers_sizes: list, weights_init: str="random", act_func_names="sigmoid"):
        self.layers = len(layers_sizes)

        if isinstance(act_func_names, str):
            self.act_func_names = (len(layers_sizes)-1) * [act_func_names]
        else:
            self.act_func_names = act_func_names

        match weights_init:
            case "random":
                self.biases = [np.random.randn(n_l, 1) for n_l in layers_sizes[1:]]
                self.weights = [
                    np.random.randn(n_l, x) for x, n_l in zip(layers_sizes[:-1], layers_sizes[1:])
                    ]
            case "xavier":
                self.biases = [np.zeros((n_l, 1)) for n_l in layers_sizes[1:]]
                self.weights = [
                    np.random.normal(loc=0, scale=np.sqrt(1/n_l), size=(n_l, x)) for x, n_l in zip(layers_sizes[:-1], layers_sizes[1:])
                    ]
            case "zeros":
                self.biases = [np.zeros((n_l, 1)) for n_l in layers_sizes[1:]]
                self.weights = [
                    np.zeros((n_l, x)) for x, n_l in zip(layers_sizes[:-1], layers_sizes[1:])
                    ]


    def fit(self, training_data: zip, validation_data: zip, epochs: int=20, batch_size: int=10, c: float=0.3, verbose: bool=False):
        """Performs neural network training using batches to modify 
        weights and biases

        Parameters
        ----------
        training_data: zip file containing pairs (x,y) of training
            observations
        validation_data: zip file containing validation data for model
            evaluation (also overfitting spotting)
        epochs: nr of epochs
        batch_size: nr of observations in every batch
        c: step size (learning rate) in gradient descent method
        verbose: if True the method will print additional info
            about the status of fitting
        """
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
        """Predicts labels for test data with feedforward"""
        test_data = list(test_data)
        preds = [np.argmax(self.feedforward(x)) for x, _ in test_data]
        return(preds)


    def feedforward(self, a: np.array):
        """Returns the output of the network"""
        for b, w, f_name in zip(self.biases, self.weights, self.act_func_names):
            a = self.activation(f_name, np.dot(w, a)+b)
        return a


    def accuracy(self, validation_data: zip, n_val: int):
        """Returns the value of one vs all accuracy criterium"""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in validation_data]
        return sum(int(x == y) for (x, y) in test_results) / n_val


    def batch_fit(self, batch: np.array, c: float):
        """Updates network's parameters by using gradient
        descent with backpropagation batch wise
        
        Parameters
        ----------
        batch: matrix containing 'batch_size' observations
        c: step size (learning rate) in gradient descent method
        """
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            delta_grad_b, delta_grad_w = self.backpropagation(x, y)
            grad_b = [gb+dgb for gb, dgb in zip(grad_b, delta_grad_b)]
            grad_w = [gw+dgw for gw, dgw in zip(grad_w, delta_grad_w)]

        self.weights = [w-c*nw for w, nw in zip(self.weights, grad_w)]
        self.biases = [b-c*nb for b, nb in zip(self.biases, grad_b)]


    def backpropagation(self, x: np.array, y: np.array):
        """Returns a tuple which contains the changes to weight
        (cost function gradient)
        
        Parameters
        ----------
        x: observation vector (of size 784 for mnist data)
        y: vector with label (for mnist we have vector of length 10
            indicating the number by position of 1)
        """
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
        """Cost function derivative"""
        return 2*(output_activations-y)


    @staticmethod
    def activation(function_name: str, x: np.array):
        """Returns a value of activation function
        
        Parameters
        ----------
        function_name: activation function for respective 
            layer
        x: matrix on which we impose activation function
        """
        match function_name:
            case "sigmoid":
                return 1.0/(1.0+np.exp(-x))
            case "relu":
                relu_func = np.vectorize(lambda x: max(0, x))
                return relu_func(x)
            case "tanh":
                return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


    def d_activation(self, function_name: str, x: float):
        """
        Returns a value of derivative of activation function

        Parameters
        ----------
        function_name: derivative of activation function for 
            respective layer
        x: matrix on which we impose the function
        """
        match function_name:
            case "sigmoid":
                y = self.activation("sigmoid", x)
                return y*(1-y)
            case "relu":
                d_relu_func = np.vectorize(lambda x: 1 if x>0 else 0)
                return d_relu_func(x)
            case "tanh":
                return 1 - self.activation("tanh", x)**2
