"""
Analyze tensor from tensor_generation and plotting.
"""
import numpy as np
import pandas as pd
import tensorly
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt
import matplotlib.cm as cm
tensorly.set_backend('numpy')

def z_score_values(A):
    '''Function that takes in the values tensor and z-scores it.'''
    B = np.zeros_like(A)
    for i in range(A.shape[2]):
        slice_face = A[:,:,i]
        mu = np.mean(slice_face)
        sigma = np.std(slice_face)
        z_scored_slice = (slice_face - mu) / sigma
        B[:,:,i] = z_scored_slice
    return B

def perform_decomposition(tensor, r):
    '''Apply z scoring and perform PARAFAC decomposition'''
    values_z = z_score_values(tensor)
    factors = parafac(values_z,rank = r, random_state=89)
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

def plot_R2X(values, n_comps):
    '''Function to plot the R2X values for various components.'''
    arr = []
    for n in range(1,n_comps):
        R2X = find_R2X(values, n)
        arr.append(R2X)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(range(1,n_comps),arr)
    plt.title('R2X for various components')
    plt.xlabel('n_components')
    plt.ylabel('R2X')
    plt.grid()
    plt.show()
    return fig

def combo_low_high(mat):
    """This function determines which combinations were high and low according to our initial conditions."""
    #First four values are IL2, IL15, IL7, IL9 that are low and the bottom 4 are their high in terms of combination values.
    IL2_low, IL2_high, IL15_low, IL15_high, IL7_low, IL7_high, IL9_low, IL9_high = [],[],[],[],[],[],[],[] #Create empty lists for each ligand to store low and high indices for each
    #lows = [[] for _ in range(4)]
    lows = [IL2_low, IL15_low, IL7_low, IL9_low]
    highs = [IL2_high, IL15_high, IL7_high, IL9_high]
    #fill up low receptor expression rates first. The indices in mat refer to the indices in combination
    for k in range(len(mat)):
        for j in range(4): #low ligand
            if mat[k,j] <= 1e0: #Condition for low ligand concentration
                lows[j].append(k)
            else: #high ligand
                highs[j].append(k)
    return lows, highs

def plot_combo_decomp(factors, mat, component_x, component_y, cell_names):
    """This function plots the combination decomposition based on high vs low receptor expression and liand concentration."""
    fig = plt.figure() #prepare figure
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    for ii in range(1, len(cell_names) + 1): #iterate over every cell
        jj = ii * len(mat)
        new_array = factors[0][jj-len(mat):jj] #repeat just the cytokine stimulation
        #deleted_factors0 = np.delete(factors[0], range(jj-len(mat),jj),0)
        plt.scatter(new_array[:,component_x - 1], new_array[:,component_y - 1], c=colors[ii-1], label = cell_names[ii-1])
    plt.xlabel('Component ' + str(component_x))
    plt.ylabel('Component ' + str(component_y))
    plt.title('Combination Decomposition Colored by Cell Type')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


def plot_low_high(factors, mat, component_x, component_y):
    """This function plots the combination decomposition based on high vs low ligand concentration."""
    lows, highs = combo_low_high(mat)
    titles = [' Combination Decomposition: Low IL2 (red) vs High IL2 (blue)', ' Combination Decomposition: Low IL15 (red) vs High IL15 (blue)', ' Combination Decomposition: Low IL7 (red) vs High IL7 (blue)', ' Combination Decomposition: Low IL9 (red) vs High IL9 (blue)']
    for cytokine in range(4):
        fig = plt.figure() #prepare figure
        plt.scatter(factors[0][lows[cytokine],component_x - 1], factors[0][lows[cytokine],component_y - 1], c="r", marker = 's', alpha = 0.2)
        plt.scatter(factors[0][highs[cytokine],component_x - 1], factors[0][highs[cytokine],component_y - 1], c = "b", marker = '^', alpha = 0.2)
        plt.xlabel('Component ' + str(component_x))
        plt.ylabel('Component ' + str(component_y))
        plt.title(titles[cytokine])
    return fig

def plot_values_decomposition(factors, component_x, component_y):
    """This function performs the values decomposition and plots it with colors separating low from high."""
    #Generate a plot for component x vs component y of the factors[2] above representing our values
    labels = ['IL2', 'IL15', 'IL7', 'IL9', 'IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #Set Active to color red. Set Surface to color blue. Set Total to color black
    for i in range(len(factors[2])):
        if i in range(4):
            c = 'r'
            if i==0:
                plt.scatter(factors[2][:,component_x - 1][i], factors[2][:,component_y - 1][i], color = c, label = 'Ligand Activity')
            else:
                plt.scatter(factors[2][:,component_x - 1][i], factors[2][:,component_y - 1][i], color = c)
            ax.annotate(labels[i], xy=(factors[2][:,component_x - 1][i], factors[2][:,component_y - 1][i]), xytext = (0, 0), textcoords = 'offset points')
        elif i in range(4,10):
            c = 'b'
            if i == 4:
                plt.scatter(factors[2][:,component_x - 1][i], factors[2][:,component_y - 1][i], color = c, label = 'Surface Receptor')
            else:
                plt.scatter(factors[2][:,component_x - 1][i], factors[2][:,component_y - 1][i], color = c)
            ax.annotate(labels[i], xy=(factors[2][:,component_x - 1][i], factors[2][:,component_y - 1][i]), xytext = (0, 0), textcoords = 'offset points')
        else:
            c = 'k'
            if i==10:
                plt.scatter(factors[2][:,component_x - 1][i], factors[2][:,component_y - 1][i], color = c, label = 'Total Receptor')
            else:
                plt.scatter(factors[2][:,component_x - 1][i], factors[2][:,component_y - 1][i], color = c)
            ax.annotate(labels[i], xy=(factors[2][:,component_x - 1][i], factors[2][:,component_y - 1][i]), xytext = (0, 0), textcoords = 'offset points')

    plt.xlabel('Component ' + str(component_x))
    plt.ylabel('Component ' + str(component_y))
    plt.title('Values decomposition')
    plt.legend()
    return fig

def plot_timepoint_decomp(factors, component_x, component_y):
    '''Function that returns the timepoint decomposition plot for the decomposed tensor.'''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(factors[1])):
        plt.scatter(factors[1][:,component_x - 1][i], factors[1][:,component_y - 1][i], color = 'k')
        if i == 999:
            ax.annotate(str(i+1), xy=(factors[1][:,component_x - 1][i], factors[1][:,component_y - 1][i]), xytext = (0, 0), textcoords = 'offset points')
    plt.xlabel('Component ' + str(component_x))
    plt.ylabel('Component ' + str(component_y))
    plt.title('Timepoint Decomposition')
    return fig

def calculate_correlation(tensor,mat,r):
    '''Make a pandas dataframe for correlation coefficients between components and initial ligand stimulation-input variables.'''
    factors = perform_decomposition(tensor, r)
    coeffs = np.zeros((factors[0].shape[1], mat.shape[1]))
    for i in range(mat.shape[1]):
        arr = []
        for j in range(factors[0].shape[1]):
            arr.append(np.corrcoef(mat[:,i], factors[0][:,j], rowvar=False)[0,1])
        coeffs[:,i] = np.array(arr)

    df = pd.DataFrame({'Component': range(1,9),'IL2': coeffs[:,0], 'IL15': coeffs[:,1], 'IL7': coeffs[:,2], 'IL9':coeffs[:,3]})
    df = df.set_index('Component')
    return df
