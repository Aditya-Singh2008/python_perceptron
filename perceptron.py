import numpy as np

class Perceptron:
    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count
        self.rng = np.random.default_rng(10)

    def passdata(self, inputs, outputs, learningrate):
        self.inputs = np.array(inputs, dtype=float)
        self.outputs = np.array(outputs, dtype=float)
        self.learningrate = learningrate

    def initialize_weights(self):
        self.weights = self.rng.random((self.output_count, self.input_count))
        self.bias = self.rng.random((self.output_count, 1))
        self.dweights = np.zeros((self.output_count, self.input_count), dtype=float)
        self.dbias = np.zeros((self.output_count, 1), dtype=float)
    
    def forwardpass(self):
        self.wsums, self.dwvdot = self.vdot(self.weights, self.inputs)
        self.wsums, self.daddbias = self.addbias(self.wsums, self.bias)
        self.wsums, self.dwsigmoid = self.sigmoid(self.wsums)
        self.mse, self.dwmse = self.meansquarederr(self.outputs, self.wsums, self.output_count)

        self.dweights = self.dwvdot * self.daddbias * self.dwsigmoid * self.dwmse
        self.dbias = np.sum(self.daddbias * self.dwsigmoid * self.dwmse, axis=1, keepdims=True)

    def backprop(self):
        self.weights -= (self.learningrate * self.dweights)
        self.bias -= (self.learningrate * self.dbias)

    def vdot(self, weights, inputs):
        dot = np.dot(weights, inputs)
        def dvdot(x):
            return np.transpose(x)
        return  dot, dvdot(inputs)
    
    def addbias(self, weights, bias):
        modified = weights * bias
        def daddbias(x):
            return np.ones_like(x)
        return modified, daddbias(self.bias)
    
    def sigmoid(self, array):
        act = (1/(1+ np.exp(-array)))
        def divsig(x):
            return (x * (1 - x))
        return  act, divsig(act)
    
    def meansquarederr(self, outputs, predicted, arrlen):
        err = np.mean((outputs - predicted) ** 2)
        def divmse(x, y, len):
            return ((2.0/len) * (x - y))
        return err, divmse(predicted, outputs, arrlen)

    def print_all(self):
        print(self.inputs)
        print(self.outputs)
        print("w: ", self.weights)
        print("b: ", self.bias)
        print(self.wsums, np.size(self.wsums))
        print("mse: ", self.mse)
        print(self.dweights)