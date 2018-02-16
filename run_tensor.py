"""
Run tensor_generation file.
"""
import os
import pickle
from ckine.tensor_generation import findy, activity_surf_tot

y_of_combinations, mat = findy(2,2)
values = activity_surf_tot(y_of_combinations)
#tens_concat = np.concatenate((np.atleast_3d(mat),values), axis = 1)

#filehandler = open("Tensor_results", 'w')
#pickle.dump((mat,values), filehandler)

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine/data/Tensor_results/Sampling.pickle")
with open(filename, 'wb') as f:
    pickle.dump((mat,values), f)