"""
Analyze tensor from tensor_generation and plotting.
"""
import numpy as np
import pandas as pd
import tensorly
from tensorly.decomposition import parafac
tensorly.set_backend('numpy')

def z_score_values(A):
    '''Function that takes in the values tensor and z-scores it.'''
    B = np.zeros_like(A)
    for i in range(A.shape[3]):
        slice_face = A[:,:,:,i]
        mu = np.mean(slice_face)
        sigma = np.std(slice_face)
        z_scored_slice = (slice_face - mu) / sigma
        B[:,:,:,i] = z_scored_slice
    return B

def perform_decomposition(tensor, r):
    '''Apply z scoring and perform PARAFAC decomposition'''
    values_z = z_score_values(tensor)
    factors = parafac(values_z, rank = r) #can do verbose and tolerance (tol)
    return factors

def find_R2X(values, n_comp):
    '''Compute R2X'''
    z_values = z_score_values(values)
    factors = perform_decomposition(z_values, n_comp)
    values_reconstructed = tensorly.kruskal_to_tensor(factors)
    return 1 - np.var(values_reconstructed - z_values) / np.var(z_values)

def combo_low_high(mat):
    """ This function determines which combinations were high and low according to our initial conditions. """
    # First six values are IL2, IL15, IL7, IL9, IL4, IL21 that are low and the bottom 6 are their high in terms of combination values.

    lows = [[] for _ in range(6)]
    highs = [[] for _ in range(6)]
    # Fill low receptor expression rates first. The indices in mat refer to the indices in combination
    for k in range(len(mat)):
        for j in range(6): #low ligand
            if mat[k,j] <= 1e0: #Condition for low ligand concentration
                lows[j].append(k)
            else: #high ligand
                highs[j].append(k)
    return lows, highs

def calculate_correlation(tensor,mat,r):
    '''Make a pandas dataframe for correlation coefficients between components and initial ligand stimulation-input variables.'''
    factors = perform_decomposition(tensor, r)
    coeffs = np.zeros((factors[0].shape[1], mat.shape[1]))
    for i in range(mat.shape[1]):
        arr = []
        for j in range(factors[0].shape[1]):
            arr.append(np.corrcoef(mat[:,i], factors[0][:,j], rowvar=False)[0,1])
        coeffs[:,i] = np.array(arr)

    df = pd.DataFrame({'Component': range(1,9),'IL2': coeffs[:,0], 'IL15': coeffs[:,1], 'IL7': coeffs[:,2], 'IL9':coeffs[:,3],'IL4':coeffs[:,4],'IL21':coeffs[:,5]})
    df = df.set_index('Component')
    return df
