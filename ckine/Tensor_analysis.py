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

def reorient_one(factors, component_index):
    """Function that takes in the 4 factor matrices and decides if that column index should flip or not and then flips it."""
    factors_idx = [factors[0][:,component_index], factors[1][:,component_index], factors[2][:,component_index], factors[3][:,component_index]]
    component_means = np.array([np.mean(np.power(factors_idx[0],3)), np.mean(np.power(factors_idx[1],3)), np.mean(np.power(factors_idx[2],3)), np.mean(np.power(factors_idx[3],3))])
    if np.sum(component_means < 0) >= 2 and np.sum(component_means < 0) < 4: #if at least 2 are negative, then flip the negative component and keep others unchanged
        count = 1
        for index, factor_idx in enumerate(factors_idx):
            if component_means[index] < 0 and count < 3:
                factors[index][:, component_index] = factor_idx * -1
                count += 1
    elif np.sum(np.array(component_means) < 0) == 4:
        for index, factor_idx in enumerate(factors_idx):
            factors[index][:,component_index] = factor_idx * -1
    return factors

def reorient_factors(factors):
    """This function is to reorient the factors if at least one component in two factors matrices are negative."""
    for jj in range(factors[0].shape[1]):
        factors = reorient_one(factors, jj)
    return factors

def find_R2X(values, factors):
    '''Compute R2X'''
    z_values = z_score_values(values)
    values_reconstructed = tensorly.kruskal_to_tensor(factors)
    return 1 - np.var(values_reconstructed - z_values) / np.var(z_values)

def R2X_remove_one(values, factors, n_comps):
    """Generate additional R2X plot for removing single components from final factorization."""
    z_values = z_score_values(values)
    LigandTensor = z_values[:,:,:, 0:6]
    SurfTensor = z_values[:,:,:, 6:14]
    TotalTensor = z_values[:,:,:, 14:22]

    R2X_singles_arr = np.zeros((4, n_comps)) #0 is ligand; 1 is surface; 2 is total; 3 is overall
    for ii in range(n_comps):
        new_factors = list()
        for jj in range(4): #4 because decomposed tensor into 4 factor matrices
            new_factors.append(np.delete(factors[jj], ii, 1))

        overall_reconstructed = tensorly.kruskal_to_tensor(new_factors)
        Ligand_reconstructed = overall_reconstructed[:,:,:,0:6]
        Surf_reconstructed = overall_reconstructed[:,:,:,6:14]
        Total_reconstructed = overall_reconstructed[:,:,:,14:22]

        R2X_singles_arr[0,ii] = 1 - np.var(Ligand_reconstructed - LigandTensor) / np.var(LigandTensor)
        R2X_singles_arr[1,ii] = 1 - np.var(Surf_reconstructed - SurfTensor) / np.var(SurfTensor)
        R2X_singles_arr[2,ii] = 1 - np.var(Total_reconstructed - TotalTensor) / np.var(TotalTensor)
        R2X_singles_arr[3,ii] = 1 - np.var(overall_reconstructed - z_values) / np.var(z_values)

    return R2X_singles_arr

def split_one_comp(values, factors):
    """Decompose and reconstruct with just one designated component, and then split tensor from both original and reconstructed to determine R2X."""
    z_values = z_score_values(values)
    R2X_array = np.zeros((4)) # An array of to store the R2X values for each split tensor and the overall one [index 3]. 
    LigandTensor = z_values[:,:,:, 0:6]
    SurfTensor = z_values[:,:,:, 6:14]
    TotalTensor = z_values[:,:,:, 14:22]
    
    values_reconstructed = tensorly.kruskal_to_tensor(factors)
    R2X_array[3] = 1 - np.var(values_reconstructed -z_values) / np.var(z_values) #overall
    
    Ligand_reconstructed = values_reconstructed[:,:,:,0:6]
    R2X_array[0] = 1 - np.var(Ligand_reconstructed - LigandTensor) / np.var(LigandTensor) #ligand

    Surf_reconstructed = values_reconstructed[:,:,:,6:14]
    R2X_array[1] = 1 - np.var(Surf_reconstructed - SurfTensor) / np.var(SurfTensor) #surface

    Total_reconstructed = values_reconstructed[:,:,:,14:22]
    R2X_array[2] = 1 - np.var(Total_reconstructed - TotalTensor) / np.var(TotalTensor) #total

    return R2X_array

def split_types_R2X(values, factors_list, n_comp):
    """Decompose and reconstruct with n components, and then split tensor from both original and reconstructed to determine R2X. n_comp here is the number of components."""
    R2X_matrix = np.zeros((n_comp,4)) # A 4 by n_comp matrix to store the R2X values for each split tensor. 

    for ii in range(n_comp):
        array = split_one_comp(values, factors_list[ii])
        R2X_matrix[ii] = array
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

def R2X_split_ligand(values, factors):
    """Determine R2X for each ligand type for one factors matrix. Follows similar procedure to split_R2X. Return a single array with R2X for each cytokine. IL2 and 15 are still combined here."""
    z_values = z_score_values(values)
    AllLigandTensors = split_values_by_ligand(z_values)
    values_reconstructed = tensorly.kruskal_to_tensor(factors)
    AllLigandReconstructed = split_values_by_ligand(values_reconstructed)
    R2X_by_ligand = np.zeros(6) #R2X at each component number with respect to each of the 6 cytokines
    for ii in range(6):
        R2X_by_ligand[ii] = 1 - np.var(AllLigandReconstructed[ii] - AllLigandTensors[ii]) / np.var(AllLigandTensors[ii])
    return R2X_by_ligand

def split_values_by_ligand(tensor):
    """Takes in tensor, either values, or values reconstructed, and returns a list of each split tensor in a list."""
    split_list = list()
    IL2Tensor =  split_list.append(tensor[:,:,:, [0,5,6,13,14]]) #IL2,IL15,IL7,IL9,IL4,IL21,IL2Ra,IL2Rb,gc,IL15Ra,IL7Ra,IL9R,IL4Ra,IL21Ra
    IL15Tensor = split_list.append(tensor[:,:,:, [0,6,8,14,16]])
    IL7Tensor = split_list.append(tensor[:,:,:, [1, 9, 17]])
    IL9Tensor = split_list.append(tensor[:,:,:, [2,10,18]])
    IL4Tensor = split_list.append(tensor[:,:,:, [3,11,19]])
    IL21Tensor = split_list.append(tensor[:,:,:, [4,12,20]])
    return split_list

def percent_reduction_by_ligand(values, factors):
    """Removing single components from final factorization (factors input) and performing percent reduction for all cytokine R2X."""
    z_values = z_score_values(values)
    AllLigandTensors = split_values_by_ligand(z_values)

    R2X_ligand_mx = np.zeros((6, factors[0].shape[1])) #0 is IL2; 1 is IL15; 2 is IL7; 3 is IL9; 4 is IL4; 5 is IL21
    
    for ii in range(factors[0].shape[1]):
        new_factors = list()
        for jj in range(4): #4 because decomposed tensor into 4 factor matrices
            new_factors.append(np.delete(factors[jj], ii, 1))
    
        overall_reconstructed = tensorly.kruskal_to_tensor(new_factors)
        AllLigandReconstructed = split_values_by_ligand(overall_reconstructed)
        for jj in range(6):
            R2X_ligand_mx[jj,ii] = 1 - np.var(AllLigandReconstructed[jj] - AllLigandTensors[jj]) / np.var(AllLigandTensors[jj])
    return R2X_ligand_mx
