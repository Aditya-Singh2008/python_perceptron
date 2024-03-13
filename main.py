import os
import numpy as np
from perceptron import *

def main():
    inp = np.arange(1, 3, dtype=float)
    out = np.arange(1, 3, dtype=float)
    ptron = Perceptron(2, 2)

    ptron.passdata(inp, out)

    ptron.initialize_weights()
    ptron.initialize_wsums()
    ptron.forwardpass()
    ptron.print_all()

main()