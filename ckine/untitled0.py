# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:57:45 2018

@author: alifa
"""

import os
import pickle
import tensorly as tl
import numpy as np
from tensorly.decomposition import partial_tucker, tucker

X = tl.tensor(np.arange(24).reshape((3, 4, 2)))
core, factors = tucker(X, ranks=[3, 4, 2])