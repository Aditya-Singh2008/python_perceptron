import os
import numpy as np
from perceptron import *

def main():
    inp = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = [1.0, 2.0, 3.0, 4.0, 5.0]
    ptron = Perceptron(1, 1, inp, out)

    ptron.initialize_weights()
    ptron.initialize_wsums()
    ptron.print_all()

main()