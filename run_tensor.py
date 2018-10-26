"""File to pickle factorization into a new file."""
import os
import numpy as np
import pickle
from ckine.tensor_generation import prepare_tensor
from ckine.Tensor_analysis import perform_decomposition, perform_tucker, find_R2X_tucker

n_ligands = 5
values, _, _, _, _ = prepare_tensor(n_ligands)

rank_list = [5,15,25,5]
out_tucker = perform_tucker(values[:,:,:,[0,1,2,3,4]], rank_list) #This contains the core tensor and the factors matrices to help in storing. 
print(find_R2X_tucker(values[:,:,:,[0,1,2,3,4]], out_tucker, subt = True))

factors_activity = []
for jj in range(6):
    factors = perform_decomposition(np.concatenate((values[:,:,:,[0,1,2,3,4]], values[:,:,:,[0,1,2,3,4]]), axis = 3), jj+1, subt = False)
    factors_activity.append(factors)

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine/data/factors_results/Sampling.pickle")
with open(filename, 'wb') as f:
    pickle.dump([factors_activity, out_tucker], f)
