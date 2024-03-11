import numpy as np
import random

class Perceptron:
    def __init__(self, input_count, output_count, inputs, eoutputs):
        self.input_count = input_count
        self.output_count = output_count
        self.inputs = np.array(inputs, dtype=float)
        self.eoutputs = np.array(eoutputs, dtype=float)
        self.rng = np.random.default_rng(2)

    def initialize_weights(self):
        self.weights = self.rng.random(self.input_count)
    def initialize_wsums(self):
        self.wsums = np.zeros(self.input_count, dtype=float)



    def print_all(self):
        print(self.inputs)
        print(self.eoutputs)
        print(self.weights)
        print(self.wsums)
