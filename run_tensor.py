"""File to pickle factorization into a new file."""
import os
import pickle
from ckine.tensor_generation import prepare_tensor
from ckine.Tensor_analysis import perform_decomposition

n_ligands = 2
values, _, _, _, _ = prepare_tensor(n_ligands)
factors_list = []
for ii in range(20):
    factors = perform_decomposition(values, ii+1)
    factors_list.append(factors)
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine/data/factors_results/Sampling.pickle")
with open(filename, 'wb') as f:
    pickle.dump((factors_list), f)