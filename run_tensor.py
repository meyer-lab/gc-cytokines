"""File to pickle factorization into a new file."""
import os
import numpy as np
import tensorly as tl
import cupy as cp
import pickle
import time
from ckine.tensor_generation import prepare_tensor
from ckine.Tensor_analysis import perform_decomposition, perform_tucker, find_R2X_tucker, tensorly_backend

n_ligands = 4
values, _, _, _, _ = prepare_tensor(n_ligands)
values = tl.tensor(values)
rank_list = [5,15,25,5]
out_tucker = perform_tucker(values[:,:,:,[0,1,2,3,4]], rank_list) #This contains the core tensor and the factors matrices to help in storing. 
print(find_R2X_tucker(values[:,:,:,[0,1,2,3,4]], out_tucker, subt = True))

factors_activity = []
for jj in range(6):
    tic = time.clock()
    print(jj)
    factors = perform_decomposition(tl.concatenate((values[:,:,:,[0,1,2,3,4]],values[:,:,:,[0,1,2,3,4]]),axis = 3) , jj+1, subt = False)
    factors_activity.append(factors)
toc = time.clock()
print(toc - tic)
    
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine/data/factors_results/Sampling.pickle")
with open(filename, 'wb') as f:
    pickle.dump([factors_activity, out_tucker], f)
