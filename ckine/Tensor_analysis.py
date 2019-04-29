"""
Analyze tensor from tensor_generation and plotting.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, non_negative_tucker
from tensorly.decomposition.candecomp_parafac import normalize_factors
from tensorly.metrics.regression import variance as tl_var

backend = 'numpy'  # Only place to choose what the backend should be. numpy = 0. cupy = 1. other backends we desire = 2, ...
tl.set_backend(backend)  # Set the backend within every file that imports from Tensor_analysis.py

# Set whether or not we subtract in one place so we're consistent
subtract = False


def z_score_values(A, cell_dim):
    ''' Function that takes in the values tensor and z-scores it. '''
    assert cell_dim < A.ndim
    convAxes = tuple([i for i in range(A.ndim) if i != cell_dim])
    convIDX = [None] * A.ndim
    convIDX[cell_dim] = slice(None)

    sigma = np.std(A, axis=convAxes)
    if subtract is False:
        return A / sigma[tuple(convIDX)]

    mu = tl.mean(A, axis=convAxes)
    return (A - mu[tuple(convIDX)]) / sigma[tuple(convIDX)]

def R2X(reconstructed, original):
    ''' Calculates R2X of two tensors. '''
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)


def perform_decomposition(tensor, r, cell_dim):
    ''' Apply z scoring and perform PARAFAC decomposition. '''
    factors = non_negative_parafac(z_score_values(tensor, cell_dim), r, tol=1.0E-9, n_iter_max=1000)
    factors, weights = normalize_factors(factors)  # Position 0 is factors. 1 is weights.
    factors[2] = factors[2] * weights[np.newaxis, :]  # Put remaining weighting in ligands
    return factors


def perform_tucker(tensor, rank_list, cell_dim):
    '''Function to peform tucker decomposition.'''
    out = non_negative_tucker(z_score_values(tensor, cell_dim), rank_list, tol=1.0E-9, n_iter_max=1000)  # index 0 is for core tensor, index 1 is for factors; out is a list of core and factors
    return out


def find_R2X_tucker(values, out, cell_dim):
    '''Compute R2X for the tucker decomposition.'''
    return R2X(tl.tucker_to_tensor(out[0], out[1]), z_score_values(values, cell_dim))


def find_R2X(values, factors, cell_dim):
    '''Compute R2X from parafac. Note that the inputs values and factors are in numpy.'''
    return R2X(tl.kruskal_to_tensor(factors), z_score_values(values, cell_dim))
