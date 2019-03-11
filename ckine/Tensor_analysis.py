"""
Analyze tensor from tensor_generation and plotting.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, non_negative_tucker
from tensorly.metrics.regression import variance as tl_var

backend = 'numpy' #Only place to choose what the backend should be. numpy = 0. cupy = 1. other backends we desire = 2, ... 
tl.set_backend(backend) #Set the backend within every file that imports from Tensor_analysis.py

# Set whether or not we subtract in one place so we're consistent
subtract = False

def z_score_values(A):
    ''' Function that takes in the values tensor and z-scores it. '''
    sigma = np.std(A, axis=(0, 2))
    mu = tl.mean(A, axis=(0, 2))
    if subtract is False:
        mu[:] = 0.0
    return (A - mu[None, :, None]) / sigma[None, :, None]


def R2X(reconstructed, original):
    ''' Calculates R2X of two tensors. '''
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)

def perform_decomposition(tensor, r):
    ''' Apply z scoring and perform PARAFAC decomposition. '''
    return non_negative_parafac(z_score_values(tensor), r, tol=1.0E-9, n_iter_max=1000)

def perform_tucker(tensor, rank_list):
    '''Function to peform tucker decomposition.'''
    out = non_negative_tucker(z_score_values(tensor), rank_list, tol=1.0E-9, n_iter_max=1000) # index 0 is for core tensor, index 1 is for factors; out is a list of core and factors
    return out

def find_R2X_tucker(values, out):
    '''Compute R2X for the tucker decomposition.'''     
    return R2X(tl.tucker_to_tensor(out[0], out[1]) , z_score_values(values))

def find_R2X(values, factors):
    '''Compute R2X. Note that the inputs values and factors are in numpy.'''
    return R2X(tl.kruskal_to_tensor(factors), z_score_values(values))

def scale_time_factors(factors, component_index):
    """Scale the timepoint factor component by dividing the mean and then in the values plot multiply the values by that same number."""
    scale_factor = tl.mean(factors[0][:, component_index])
    factors[2][:, component_index] *= scale_factor
    factors[0][:, component_index] /= scale_factor
    return factors

def scale_all(factors):
    """Function to rescale all components. Timepoint factor matrix and values factor matrix."""
    for ii in range(factors[0].shape[1]):
        factors = scale_time_factors(factors, ii)
    return factors
