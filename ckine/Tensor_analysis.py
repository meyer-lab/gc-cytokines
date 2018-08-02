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

def find_R2X(values, factors):
    '''Compute R2X'''
    z_values = z_score_values(values)
    values_reconstructed = tensorly.kruskal_to_tensor(factors)
    return 1 - np.var(values_reconstructed - z_values) / np.var(z_values)

def R2X_singles(values, factors_list, n_comps):
    """Generate additional R2X plot for removing single components from final factorization."""
    z_values = z_score_values(values)
    LigandTensor = z_values[:,:,:, 0:6]
    SurfTensor = z_values[:,:,:, 6:14]
    TotalTensor = z_values[:,:,:, 14:22]
    
    factors = factors_list[-1]
    R2X_singles_matrix = np.zeros((4,n_comps)) #1st row is overall R2X; 2nd row is ligand activity R2X; 3rd is Surface receptor; 4th is total receptor
    for ii in range(n_comps):
        new_factors = list()
        for jj in range(4): #4 because decomposed tensor into 4 factor matrices
            new_factors.append(np.delete(factors[jj], ii, 1))

        overall_reconstructed = tensorly.kruskal_to_tensor(new_factors)
        Ligand_reconstructed = overall_reconstructed[:,:,:,0:6]
        Surf_reconstructed = overall_reconstructed[:,:,:,6:14]
        Total_reconstructed = overall_reconstructed[:,:,:,14:22]
        R2X_singles_matrix[0,ii] = 1 - np.var(overall_reconstructed - z_values) / np.var(z_values)
        R2X_singles_matrix[1,ii] = 1 - np.var(Ligand_reconstructed - LigandTensor) / np.var(LigandTensor)
        R2X_singles_matrix[2,ii] = 1 - np.var(Surf_reconstructed - SurfTensor) / np.var(SurfTensor)
        R2X_singles_matrix[3, ii] = 1 - np.var(Total_reconstructed - TotalTensor) / np.var(TotalTensor)
    return R2X_singles_matrix

def split_R2X(values, factors_list, n_comp):
    """Decompose and reconstruct with n components, and then split tensor from both original and reconstructed to determine R2X."""
    z_values = z_score_values(values)
    R2X_matrix = np.zeros((3,n_comp)) # A 3 by n_comp matrix to store the R2X values for each split tensor. 
    LigandTensor = z_values[:,:,:, 0:6]
    SurfTensor = z_values[:,:,:, 6:14]
    TotalTensor = z_values[:,:,:, 14:22]

    for ii in range(n_comp):
        factors = factors_list[ii]
        values_reconstructed = tensorly.kruskal_to_tensor(factors)
        
        Ligand_reconstructed = values_reconstructed[:,:,:,0:6]
        R2X_matrix[0,ii] = 1 - np.var(Ligand_reconstructed - LigandTensor) / np.var(LigandTensor)
        
        Surf_reconstructed = values_reconstructed[:,:,:,6:14]
        R2X_matrix[1,ii] = 1 - np.var(Surf_reconstructed - SurfTensor) / np.var(SurfTensor)
        
        Total_reconstructed = values_reconstructed[:,:,:,14:22]
        R2X_matrix[2, ii] = 1 - np.var(Total_reconstructed - TotalTensor) / np.var(TotalTensor)

    return R2X_matrix

def combo_low_high(mat, lig):
    """ This function determines which combinations were high and low according to our initial conditions. """
    # First six values are IL2, IL15, IL7, IL9, IL4, IL21 that are low and the bottom 6 are their high in terms of combination values.
    ILs = np.logspace(-3, 2, num=lig)
    
    IL2_low_high = [[] for _ in range(len(ILs))]
    IL15_low_high = [[] for _ in range(len(ILs))]
    
    #lows = [[] for _ in range(2)]
    #highs = [[] for _ in range(2)]
    # Fill low receptor expression rates first. The indices in mat refer to the indices in combination
    for a in range(len(ILs)):
        for k in range(len(mat)):
            if mat[k,0] == ILs[a]: #Condition for low ligand concentration
                IL2_low_high[a].append(k)
            if mat[k,1] == ILs[a]:
                IL15_low_high[a].append(k)
    return IL2_low_high, IL15_low_high

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
