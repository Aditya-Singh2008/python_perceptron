import numpy as np
import random

class Perceptron:
    def __init__(self, input_count, output_count):
        self.input_count = input_count

    def initialize_weights(self):
        self.weights = [random.random()] * self.input_count
    def print_weights(self):
        print(self.weights[0])
        