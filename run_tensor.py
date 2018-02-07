"""
Run tensor generation_file.
"""
from ckine.tensor_generation import findy, activity_surf_tot

y_of_combinations = findy()
values = activity_surf_tot(y_of_combinations)