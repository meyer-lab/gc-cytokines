"""
Analyze tensor from tensor_generation and plotting.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, non_negative_tucker
from tensorly.decomposition.candecomp_parafac import normalize_factors
from tensorly.metrics.regression import variance as tl_var

backend = 'numpy'  # Tensorly backend choice
tl.set_backend(backend)  # Set the backend within every file that imports from Tensor_analysis.py


def z_score_values(A, cell_dim):
    ''' Function that takes in the values tensor and z-scores it. '''
    assert cell_dim < tl.ndim(A)
    convAxes = tuple([i for i in range(tl.ndim(A)) if i != cell_dim])
    convIDX = [None] * tl.ndim(A)
    convIDX[cell_dim] = slice(None)

    sigma = tl.tensor(np.std(tl.to_numpy(A), axis=convAxes))
    return A / sigma[tuple(convIDX)]


def R2X(reconstructed, original):
    ''' Calculates R2X of two tensors. '''
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)


def perform_decomposition(tensor, r, weightFactor=2):
    ''' Apply z-scoring and perform PARAFAC decomposition. '''
    factors = non_negative_parafac(tensor, r, tol=1.0E-8, n_iter_max=10000)
    factors, weights = normalize_factors(factors)  # Position 0 is factors. 1 is weights.
    factors[weightFactor] *= weights[np.newaxis, :]  # Put weighting in designated factor
    return factors


def perform_tucker(tensor, rank_list):
    '''Function to peform tucker decomposition.'''
    # index 0 is for core tensor, index 1 is for factors; out is a list of core and factors
    return non_negative_tucker(tensor, rank_list, tol=1.0E-8, n_iter_max=10000)


def find_R2X_tucker(values, out):
    '''Compute R2X for the tucker decomposition.'''
    return R2X(tl.tucker_to_tensor(out[0], out[1]), values)


def find_R2X(values, factors):
    '''Compute R2X from parafac. Note that the inputs values and factors are in numpy.'''
    return R2X(tl.kruskal_to_tensor(factors), values)
