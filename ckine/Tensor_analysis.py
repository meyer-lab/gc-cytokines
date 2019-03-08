"""
Analyze tensor from tensor_generation and plotting.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker

def tensorly_backend(bknd):
    '''Function to convert back and forth between numpy and cupy backends. Always works with numpy unless set as False which switches to cupy.'''
    if bknd == 0:
        tl.set_backend('numpy')
    elif bknd == 1:
        tl.set_backend('tensorflow')

backend = 0 #Only place to choose what the backend should be. numpy = 0. cupy = 1. other backends we desire = 2, ... 
tensorly_backend(bknd = backend) #Set the backend within every file that imports from Tensor_analysis.py

def tl_var(matrix):
    '''Function to compute the variance of a matrix using the basic wrapper functions within tensorly. Tensotly does not have variance formula.'''
    mean = tl.mean(matrix)
    var = tl.sum((matrix - mean)**2) / matrix.size
    return var

def z_score_values(A, subtract = True):
    '''Function that takes in the values tensor and z-scores it.'''
    B = tl.zeros_like(A)
    for i in range(A.shape[3]):
        slice_face = A[:,:,:,i]
        mu = tl.mean(slice_face)
        sigma = tl.sqrt(tl_var(slice_face))
        if subtract is True:
            z_scored_slice = (slice_face - mu) / sigma
        elif subtract is False:
            z_scored_slice = slice_face / sigma
        B[:,:,:,i] = z_scored_slice
    return B

def perform_decomposition(tensor, r, subt = True):
    '''Apply z scoring and perform PARAFAC decomposition'''
    values_z = z_score_values(tensor, subtract = subt)
    factors = parafac(values_z, rank = r) #can do verbose and tolerance (tol)
    return factors

def perform_tucker(tensor, rank_list, subt = True):
    '''Function to peform tucker decomposition.'''
    values_z = z_score_values(tensor, subtract = subt)
    out = tucker(values_z, ranks = rank_list, init = 'random') #index 0 is for core tensor, index 1 is for factors; out is a list of core and factors
    return out

def find_R2X_tucker(values, out, subt = True):
    '''Compute R2X for the tucker decomposition.'''
    z_values = z_score_values(values, subtract = subt)
    values_reconstructed = tl.tucker_to_tensor(out[0], out[1])        
    return 1 - tl_var(values_reconstructed - z_values) / tl_var(z_values)

def reorient_one(factors, component_index):
    """Function that takes in the 4 factor matrices and decides if that column index should flip or not and then flips it."""
    factors_idx = [factors[0][:,component_index], factors[1][:,component_index], factors[2][:,component_index], factors[3][:,component_index]]
    component_means = tl.tensor([tl.mean(factors_idx[0]**3), tl.mean(factors_idx[1]**3), tl.mean(factors_idx[2]**3), tl.mean(factors_idx[3]**3)])
    if tl.sum(component_means < 0) >= 2 and tl.sum(component_means < 0) < 4: #if at least 2 are negative, then flip the negative component and keep others unchanged
        count = 1
        for index, factor_idx in enumerate(factors_idx):
            if component_means[index] < 0 and count < 3:
                factors[index][:, component_index] = factor_idx * -1
                count += 1
    elif tl.sum(tl.tensor(component_means) < 0) == 4:
        for index, factor_idx in enumerate(factors_idx):
            factors[index][:,component_index] = factor_idx * -1
    return factors

def reorient_factors(factors):
    """This function is to reorient the factors if at least one component in two factors matrices are negative."""
    for jj in range(factors[0].shape[1]):
        factors = reorient_one(factors, jj)
    return factors

def find_R2X(values, factors, subt = True):
    '''Compute R2X. Note that the inputs values and factors are in numpy.'''
    z_values = z_score_values(values, subtract = subt)
    values_reconstructed = tl.kruskal_to_tensor(factors)
    return 1 - tl_var(values_reconstructed - z_values) / tl_var(z_values)

def R2X_remove_one(values, factors, n_comps):
    """Generate additional R2X plot for removing single components from final factorization."""
    z_values = z_score_values(values)
    LigandTensor = z_values[:,:,:, 0:5]

    R2X_singles_arr = tl.zeros((2, n_comps)) #0 is ligand; 1 is overall
    for ii in range(n_comps):
        new_factors = list()
        for jj in range(4): #4 because decomposed tensor into 4 factor matrices
            new_factors.append(tl.tensor(np.delete(tl.to_numpy(factors[jj], ii, 1))))

        overall_reconstructed = tl.kruskal_to_tensor(new_factors)
        Ligand_reconstructed = overall_reconstructed[:,:,:,0:5]

        R2X_singles_arr[0,ii] = 1 - tl_var(Ligand_reconstructed - LigandTensor) / tl_var(LigandTensor)
        R2X_singles_arr[1,ii] = 1 - tl_var(overall_reconstructed - z_values) / tl_var(z_values)

    return R2X_singles_arr

def split_one_comp(values, factors):
    """Decompose and reconstruct with just one designated component, and then split tensor from both original and reconstructed to determine R2X."""
    z_values = z_score_values(values)
    R2X_array = tl.zeros((2)) # An array of to store the R2X values for each split tensor and the overall one [index 3].
    LigandTensor = z_values[:,:,:, 0:5]

    values_reconstructed = tl.kruskal_to_tensor(factors)
    R2X_array[1] = 1 - tl_var(values_reconstructed -z_values) / tl_var(z_values) #overall

    Ligand_reconstructed = values_reconstructed[:,:,:,0:5]
    R2X_array[0] = 1 - tl_var(Ligand_reconstructed - LigandTensor) / tl_var(LigandTensor) #ligand
    return R2X_array

def split_types_R2X(values, factors_list, n_comp):
    """Decompose and reconstruct with n components, and then split tensor from both original and reconstructed to determine R2X. n_comp here is the number of components."""
    R2X_matrix = tl.zeros((n_comp,2)) # A 4 by n_comp matrix to store the R2X values for each split tensor.

    for ii in range(n_comp):
        array = split_one_comp(values, factors_list[ii])
        R2X_matrix[ii] = array
    return R2X_matrix

def R2X_split_ligand(values, factors):
    """Determine R2X for each ligand type for one factors matrix. Follows similar procedure to split_R2X. Return a single array with R2X for each cytokine. IL2 and 15 are still combined here."""
    z_values = z_score_values(values)
    AllLigandTensors = split_values_by_ligand(z_values)
    values_reconstructed = tl.kruskal_to_tensor(factors)
    AllLigandReconstructed = split_values_by_ligand(values_reconstructed)
    R2X_by_ligand = tl.zeros(5) #R2X at each component number with respect to each of the 6 cytokines
    for ii in range(5):
        R2X_by_ligand[ii] = 1 - tl_var(AllLigandReconstructed[ii] - AllLigandTensors[ii]) / tl_var(AllLigandTensors[ii])
    return R2X_by_ligand

def split_values_by_ligand(tensor):
    """Takes in tensor, either values, or values reconstructed, and returns a list of each split tensor in a list. """
    return tl.tensor(np.split(tl.to_numpy(tensor), tensor.shape[3], axis=3))

def percent_reduction_by_ligand(values, factors):
    """Removing single components from final factorization (factors input) and performing percent reduction for all cytokine R2X."""
    z_values = z_score_values(values)
    AllLigandTensors = split_values_by_ligand(z_values)

    R2X_ligand_mx = tl.zeros((5, factors[0].shape[1])) #0 is IL2 and IL15 Combined; 1 is IL7; 2 is IL9; 3 is IL4; 4 is IL21

    for ii in range(factors[0].shape[1]):
        new_factors = list()
        for jj in range(4): #4 because decomposed tensor into 4 factor matrices
            new_factors.append(tl.tensor(np.delete(tl.to_numpy(factors[jj]), ii, 1)))

        overall_reconstructed = tl.kruskal_to_tensor(new_factors)
        AllLigandReconstructed = split_values_by_ligand(overall_reconstructed)
        for jj in range(5):
            R2X_ligand_mx[jj,ii] = 1 - tl_var(AllLigandReconstructed[jj] - AllLigandTensors[jj]) / tl_var(AllLigandTensors[jj])
    return R2X_ligand_mx

def scale_time_factors(factors, component_index):
    """Scale the timepoint factor component by dividing the mean and then in the values plot multiply the values by that same number."""
    scale_factor = tl.mean(factors[0][:, component_index])
    factors[3][:, component_index] *= scale_factor
    factors[0][:, component_index] /= scale_factor
    return factors

def scale_all(factors):
    """Function to rescale all components. Timepoint factor matrix and values factor matrix."""
    for ii in range(factors[0].shape[1]):
        factors = scale_time_factors(factors, ii)
    return factors
