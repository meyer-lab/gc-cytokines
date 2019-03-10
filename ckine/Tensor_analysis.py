"""
Analyze tensor from tensor_generation and plotting.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.metrics.regression import variance as tl_var, standard_deviation

def tensorly_backend(bknd):
    '''Function to convert back and forth between numpy and cupy backends. Always works with numpy unless set as False which switches to cupy.'''
    if bknd == 0:
        tl.set_backend('numpy')
    elif bknd == 1:
        tl.set_backend('cupy')

backend = 0 #Only place to choose what the backend should be. numpy = 0. cupy = 1. other backends we desire = 2, ... 
tensorly_backend(bknd = backend) #Set the backend within every file that imports from Tensor_analysis.py

def z_score_values(A, subtract):
    ''' Function that takes in the values tensor and z-scores it. '''
    sigma = np.std(A, axis=(0, 1))
    mu = tl.mean(A, axis=(0, 1))
    if subtract is False:
        mu[:] = 0.0
    return (A - mu[None, None, :]) / sigma[None, None, :]


def R2X(reconstructed, original):
    ''' Calculates R2X of two tensors. '''
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)

def perform_decomposition(tensor, r, subt):
    '''Apply z scoring and perform PARAFAC decomposition'''
    values_z = z_score_values(tensor, subtract = subt)
    factors = parafac(values_z, rank = r) # can do verbose and tolerance (tol)
    return factors

def perform_tucker(tensor, rank_list, subt):
    '''Function to peform tucker decomposition.'''
    values_z = z_score_values(tensor, subtract=subt)
    out = tucker(values_z, ranks=rank_list) # index 0 is for core tensor, index 1 is for factors; out is a list of core and factors
    return out

def find_R2X_tucker(values, out, subt):
    '''Compute R2X for the tucker decomposition.'''
    z_values = z_score_values(values, subtract = subt)
    values_reconstructed = tl.tucker_to_tensor(out[0], out[1])        
    return R2X(values_reconstructed, z_values)

def reorient_one(factors, component_index):
    """Function that takes in the 4 factor matrices and decides if that column index should flip or not and then flips it."""
    factors_idx = [factors[0][:,component_index], factors[1][:,component_index], factors[2][:,component_index]]
    component_means = tl.tensor([tl.mean(factors_idx[0]**3), tl.mean(factors_idx[1]**3), tl.mean(factors_idx[2]**3)])
    if tl.sum(component_means < 0) >= 2 and tl.sum(component_means < 0) <= 3: #if at least 2 are negative, then flip the negative component and keep others unchanged
        count = 1
        for index, factor_idx in enumerate(factors_idx):
            if component_means[index] < 0 and count < 3:
                factors[index][:, component_index] = factor_idx * -1
                count += 1
    return factors

def reorient_factors(factors):
    """This function is to reorient the factors if at least one component in two factors matrices are negative."""
    for jj in range(factors[0].shape[1]):
        factors = reorient_one(factors, jj)
    return factors

def find_R2X(values, factors, subt):
    '''Compute R2X. Note that the inputs values and factors are in numpy.'''
    z_values = z_score_values(values, subtract = subt)
    values_reconstructed = tl.kruskal_to_tensor(factors)
    return R2X(values_reconstructed, z_values)

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
