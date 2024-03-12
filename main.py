import os
import numpy as np
from perceptron import *

def main():
    inp = np.arange(1, 5, dtype=float)
    out = np.arange(1, 5, dtype=float)
    ptron = Perceptron(1, 1)

    ptron.passdata(inp, out)

    ptron.initialize_weights()
    ptron.initialize_wsums()
    ptron.print_all()

main()