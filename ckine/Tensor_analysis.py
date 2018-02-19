"""
Analyze tensor
"""
import tensorly as tl
import numpy as np
from tensorly.decomposition import tucker
X = tl.tensor(np.arange(24).reshape((3, 4, 2)))
core, factors = tucker(X, ranks=[3, 4, 2])
full_tensor = tl.tucker_to_tensor(core, factors)
