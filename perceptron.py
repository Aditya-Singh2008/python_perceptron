import numpy as np

class Perceptron:
    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count
        self.rng = np.random.default_rng(2)

    def passdata(self, inputs, eoutputs):
        self.inputs = np.array(inputs, dtype=float)
        self.eoutputs = np.array(eoutputs, dtype=float)

    def initialize_weights(self):
        self.weights = self.rng.random((self.input_count, self.output_count))
    
    def initialize_wsums(self):
        self.wsums = np.zeros(self.input_count, dtype=float)
    
    def forwardpass(self):
        self.wsums = self.dotprod(self.inputs, self.weights)
    
    def dotprod(self):
        return np.dot(self.weights, self.inputs)
    

    def print_all(self):
        print(self.inputs)
        print(self.eoutputs)
        print(self.weights)
        print(self.wsums)
