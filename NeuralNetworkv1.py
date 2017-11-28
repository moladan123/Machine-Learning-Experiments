import numpy as np

# one dimensional vectors
nodes = []
# two dimensional arrays
# This is what needs to be trained and what needs to be
weights = []

# given one training example computes the error with
# the current weights
def error(data, expected):
    for i in range(len(nodes) - 1):
        nodes[i + 1] = nodes[i] * weights[i]