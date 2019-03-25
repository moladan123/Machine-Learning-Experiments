import numpy as np
import time

class Functions:
    """ A set of activation functions to choose from """

    @staticmethod
    def linear(x, derivative=False):
        if derivative:
            return 1
        else:
            return x

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            f = Functions.sigmoid
            return f(x) * (1 - f(x))
        else:
            return 1 / (1 + np.exp(-x))


class NeuralNetwork:

    def __init__(self, sizes, learning_rate=0.01):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.nodes = [np.random.rand(size) for size in self.sizes]
        self.weights = [np.random.rand(self.sizes[i], self.sizes[i + 1]) for i in range(len(self.sizes) - 1)]
        self.biases = [np.zeros(size) for size in self.sizes[1:]]

        l = Functions.linear   # last layer does not need an activation
        f = Functions.sigmoid
        self.activations= [f, f, f, l]

    def forward(self, x: np.array):
        """
        Passes in a training example for the given network
        Updates the network nodes with the new values from the training example

        @:return the last layer of the network as vector
        """
        self.nodes[0] = x

        num_edges = len(self.weights)
        for edge in range(num_edges):
            w = self.weights[edge]
            f = self.activations[edge]
            b = self.biases[edge]

            # calculate the value of the next layer
            print(w.shape, x.shape, b.shape)
            x = f(x.dot(w) + b)

            # remember the value of the edge for gradient descent
            self.nodes[edge] = x
        return x


    def backward(self, yhat):

        gradients = dict()
        prev_error = 2 * (yhat - self.nodes[-1])

        # calculate derivative at each layer
        for layer in reversed(range(len(self.activations))):
            f = self.activations[layer]
            print(prev_error, self.weights[layer], self.nodes[layer], f.__name__)
            prev_error = np.dot(prev_error, self.weights[layer]) * f(self.nodes[layer], derivative=True)
            gradients[layer] = np.dot(self.nodes[layer].T, prev_error)

        # actually update all of the weights
        print(gradients)
        for layer in range(len(self.activations)):
            self.weights[layer] -= self.learning_rate * gradients[layer]


if __name__ == "__main__":
    # creates a neural network with 5 inputs, 3 hidden layers, and 1 output node
    nn = NeuralNetwork([5, 10, 10, 10, 1])
    x = np.array([1, 2, 3, 4, 5])
    y = nn.forward(x)
    nn.backward(y)
    print(y)
    print(nn.activations)
    print(nn.nodes)
