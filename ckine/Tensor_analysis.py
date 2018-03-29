"""
Analyze tensor from Sampling.pickle and plotting.
"""
import os
import pickle
import numpy as np
import tensorly
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt
tensorly.set_backend('numpy')

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data/Tensor_results/Sampling.pickle")
with open(filename, 'rb') as file:
    sample = pickle.load(file)

mat, values = sample[0], sample[1]

def z_score_values(A):
    '''Function that takes in the values tensor and z-scores it.'''
    B = np.zeros_like(A)
    for i in range(A.shape[2]):
        slice = A[:,:,i]
        mu = np.mean(slice)
        sigma = np.std(slice)
        z_scored_slice = (slice - mu)/sigma
        B[:,:,i] = z_scored_slice
    return B

def perform_decomposition(tensor, r):
    '''Apply z scoring and perform PARAFAC decomposition'''
    values_z = z_score_values(tensor)
    factors = parafac(values_z,rank = r, random_state=93)
    return factors

def find_R2X(values, n_comp):
    '''Compute R2X'''
    factors = perform_decomposition(values , n_comp)
    values_reconstructed = tensorly.kruskal_to_tensor(factors)
    z_values = z_score_values(values)
    denominator = np.var(z_values)
    numerator = np.var(values_reconstructed - z_values)
    R2X = 1 - numerator / denominator
    return R2X


def combo_low_high(mat):
    """This function determines which combinations were high and low according to our condition."""
    #First ten values are IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL2, IL15, IL7, IL9 that are low and the bottom 10 are their high in terms of combination values
    low_high = [[] for _ in range(20)]
    #fill up low receptor expression rates first. The indices in mat refer to the indices in combination
    for k in range(len(mat)):
        for j in range(4): #low ligand
            if mat[k,j] <= 1e-3: #Condition for low ligand concentration
                low_high[j].append(k)
            else: #high ligand
                low_high[j+10].append(k)
        for i in range(4,10):
            if mat[k,i] <= 1e-3: #Condition for low receptor expression rate
                low_high[i].append(k)
            else:
                low_high[i+10].append(k)
    new_low_high = np.array(low_high)

    return new_low_high

def plot_values_decomposition(factors):
    """This function performs the values decomposition and plots it with colors separating low from high."""
    #Generate a plot for component 1 vs component 2 of the factors[2] above representing our values
    labels = ['IL2', 'IL15', 'IL7', 'IL9', 'IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #Set Active to color red
    #Set Surface to color blue
    # Set Total to color black
    for i in range(len(factors[2])):
        if i in range(4):
            c = 'r'
            if i==0:
                plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c, label = 'Ligand Activity')
            else:
                plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c)
            ax.annotate(labels[i], xy=(factors[2][:,0][i], factors[2][:,1][i]), xytext = (0, 0), textcoords = 'offset points')
        elif i in range(4,10):
            c = 'b'
            if i == 4:
                plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c, label = 'Surface Receptor')
            else:
                plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c)
            ax.annotate(labels[i], xy=(factors[2][:,0][i], factors[2][:,1][i]), xytext = (0, 0), textcoords = 'offset points')
        else:
            c = 'k'
            if i==10:
                plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c, label = 'Total Receptor')
            else:
                plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c)
            ax.annotate(labels[i], xy=(factors[2][:,0][i], factors[2][:,1][i]), xytext = (0, 0), textcoords = 'offset points')

    plt.xlabel('Component One')
    plt.ylabel('Component Two')
    plt.title('Values decomposition')
    plt.legend()
    return fig

def plot_timepoint_decomp(factors):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(factors[1])):
        plt.scatter(factors[1][:,0][i], factors[1][:,1][i], color = 'k')
        #ax.annotate(str(i+1), xy=(factors[1][:,0][i], factors[1][:,1][i]), xytext = (0, 0), textcoords = 'offset points')
        if i == 99:
            ax.annotate(str(i+1), xy=(factors[1][:,0][i], factors[1][:,1][i]), xytext = (0, 0), textcoords = 'offset points')
    plt.xlabel('Component One')
    plt.ylabel('Component Two')
    plt.title('Timepoint Decomposition')
    return fig

def plot_combo_decomp(factors, mat):
    """This function plots the combination decomposition based on high vs low receptor expression and liand concentration."""
    low_highs = combo_low_high(mat)
    titles = ['Combination Decomposition: Low IL2 (red) vs High IL2 (blue)', 'Combination Decomposition: Low IL15 (red) vs High IL15 (blue)', 'Combination Decomposition: Low IL7 (red) vs High IL7 (blue)', 'Combination Decomposition: Low IL9 (red) vs High IL9 (blue)','Combination Decomposition: Low IL2Ra (red) vs High IL2Ra (blue)', 'Combination Decomposition: Low IL2Rb (red) vs High IL2Rb (blue)', 'Combination Decomposition: Low gc (red) vs High gc (blue)', 'Combination Decomposition: Low IL15Ra (red) vs High IL15Ra (blue)', 'Combination Decomposition: Low IL7Ra (red) vs High IL7Ra (blue)','Combination Decomposition: Low IL9R (red) vs High IL9R (blue)']
    for i in range(10):
        fig = plt.figure()
        for j in low_highs[i,:]:
            j = int(j)
            plt.scatter(factors[0][:,0][j], factors[0][:,1][j], color = 'r', alpha = 0.2)
        for j in low_highs[i+10,:]:
            j = int(j)
            plt.scatter(factors[0][:,0][j], factors[0][:,1][j], color = 'b', alpha = 0.2)
        plt.xlabel('Component One')
        plt.ylabel('Component Two')
        plt.title(titles[i])
    return fig
