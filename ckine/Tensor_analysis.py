"""
Analyze tensor from Sampling.pickle
"""
import os
import pickle
import tensorly as tl
import numpy as np
from tensorly.decomposition import partial_tucker


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data/Tensor_results/Sampling.pickle")
with open(filename, 'rb') as file:
    sample = pickle.load(file)

mat, values = sample[0], sample[1]

#X = tl.tensor(values)
core1, factors1 = partial_tucker(values,[0,1,2], ranks=[1024, 100, 16])
