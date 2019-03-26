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

    def __str__(self):
        ret = "Network with sizes " + str(self.sizes) + "\n"
        ret += "learning rate " + str(self.learning_rate) + "\n"
        ret += "nodes " + str(self.nodes) + "\n"
        ret += "weights " + str(self.weights) + "\n"
        ret += "biases " + str(self.biases) + "\n"
        ret += "activations " + str([f.__name__ for f in self.activations]) + "\n"
        return ret

    def forward(self, x: np.array, verbose=False):
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
            if verbose:
                print(w.shape, x.shape, b.shape)
            x = f(x.dot(w) + b)

            # remember the value of the edge for gradient descent
            self.nodes[edge] = x
        return x

    def backward(self, yhat, verbose=False):
        """ Doesn't yet work don't use
        """

        layer_to_gradient = dict()
        prev_error = 2 * (yhat - self.nodes[-1])

        # calculate derivative at each layer
        for layer in reversed(range(len(self.activations))):
            f = self.activations[layer]

            if verbose:
                print(prev_error, self.weights[layer], self.nodes[layer], f.__name__)

            temp = np.dot(self.weights[layer], prev_error)
            prev_error = temp * f(self.nodes[layer - 1], derivative=True)

            layer_to_gradient[layer] = np.dot(self.nodes[layer].T, prev_error)

        # actually update all of the weights
        print(layer_to_gradient)
        time.sleep(5)
        for layer in range(len(self.activations)):
            self.weights[layer] -= self.learning_rate * layer_to_gradient[layer]


if __name__ == "__main__":
    # creates a neural network with 5 inputs, 3 hidden layers, and 1 output node
    nn = NeuralNetwork([5, 10, 10, 10, 1])
    x = np.array([0, 1, 0, 1, 0])
    y = nn.forward(x)
    print(nn)
    print(y)

    time.sleep(1)
    nn.backward(np.array([0]), verbose=True)
