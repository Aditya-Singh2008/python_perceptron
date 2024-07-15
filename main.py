import os
import numpy as np
from perceptron import *

def main():
    learningrate = 0.1
    inp = np.arange(1, 3, dtype=float)
    out = np.arange(1, 3, dtype=float)
    ptron = Perceptron(1, 1)

    ptron.initialize_weights()
    ptron.passdata(inp, out, learningrate)

    for i in range(5):
        ptron.forwardpass()
        ptron.print_all()
        ptron.backprop()


main()