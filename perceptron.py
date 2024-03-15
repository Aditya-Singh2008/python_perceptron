import numpy as np

class Perceptron:
    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count
        self.rng = np.random.default_rng(10)

    def passdata(self, inputs, eoutputs):
        self.inputs = np.array(inputs, dtype=float)
        self.eoutputs = np.array(eoutputs, dtype=float)

    def initialize_weights(self):
        self.weights = self.rng.random((self.output_count, self.input_count))
        self.dweights = np.zeros((self.output_count, self.input_count), dtype=float)
        print(self.dweights)

    
    def forwardpass(self):
        self.wsums = np.dot(self.weights, self.inputs)
        self.wsums, self.dweights = self.sigmoid(self.wsums)
    
    def sigmoid(self, array):
        act = (1/(1+ np.exp(-array)))
        def divsig(x):
            return (x * (1 - x))
        return  act, divsig(act)

    def print_all(self):
        print(self.inputs)
        print(self.eoutputs)
        print(self.weights)
        print(self.wsums, np.size(self.wsums))
        print(self.dweights)
