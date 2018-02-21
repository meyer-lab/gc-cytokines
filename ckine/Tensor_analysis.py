"""
Analyze tensor from Sampling.pickle
"""
import os
import pickle
import tensorly as tl
import numpy as np
from tensorly.decomposition import partial_tucker, tucker, parafac


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data/Tensor_results/Sampling.pickle")
with open(filename, 'rb') as file:
    sample = pickle.load(file)

mat, values = sample[0], sample[1]

factors = parafac(values,rank = 2)
