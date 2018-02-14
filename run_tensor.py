"""
Run tensor generation_file.
"""
import numpy as np
import pickle
from ckine.tensor_generation import findy, activity_surf_tot

y_of_combinations, mat = findy(3,3)
values = activity_surf_tot(y_of_combinations)
tens_concat = np.concatenate((mat,values), axis = 1)

filehandler = open("Tensor_results", 'w')
pickle.dump(tens_concat, filehandler)
