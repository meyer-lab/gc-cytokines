"""
This creates Figure 3.
"""
from sklearn.decomposition.pca import PCA
import matplotlib.cm as cm
from ..tensor_generation import prepare_tensor
from .figureCommon import subplotLabel, getSetup
from ..Tensor_analysis import find_R2X, split_one_comp, split_types_R2X, R2X_remove_one, percent_reduction_by_ligand, R2X_split_ligand, reorient_factors
import string
import os
import pickle
from sklearn.decomposition.pca import PCA
from .figureCommon import subplotLabel, getSetup
from ..Tensor_analysis import find_R2X, percent_reduction_by_ligand, R2X_split_ligand, reorient_factors
import numpy as np, pandas as pds
from scipy import stats
import tensorly
tensorly.set_backend('numpy')

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 3, 4
    ax, f = getSetup((16, 14), (x, y))

    # Blank out for the cartoon
    ax[0].axis('off')

    fileDir = os.path.dirname(os.path.realpath('__file__'))

    factors_filename = os.path.join(fileDir, './ckine/data/factors_results/Sampling.pickle')
    factors_filename = os.path.abspath(os.path.realpath(factors_filename))

    expr_filename = os.path.join(fileDir, './ckine/data/expr_table.csv')
    data = pds.read_csv(expr_filename) # Every column in the data represents a specific cell
    cell_names = data.columns.values.tolist()[1::] #returns the cell names from the pandas dataframe (which came from csv)
    numpy_data = data.values
    Receptor_data = np.delete(numpy_data, 0, 1)

    with open(factors_filename,'rb') as ff:
        two_files = pickle.load(ff)
    factors_activity = two_files[0] #Only the activities tensor (without surface and total receptors)

    factors = factors_activity[5]
    factors = reorient_factors(factors)

    values = tensorly.tucker_to_tensor(two_files[1][0], two_files[1][1]) #This reconstructs our values tensor from the decomposed one that we used to store our data in.
    values = np.concatenate((values, values), axis = 3)
    n_comps = 5
    factors_activ = factors_activity[n_comps]
    newfactors_activ = reorient_factors(factors_activ)

    PCA_receptor(ax[1], ax[2], cell_names, Receptor_data)
    plot_R2X(ax[3], values, factors_activity, n_comps = 6)
    plot_reduction_ligand(ax[4], values, newfactors_activ)

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    #f.tight_layout()

    return f

def plot_reduction_ligand(ax, values, factors):
    """Function to plot the percent by reduction in R2X for each ligand type."""
    old_R2X = R2X_split_ligand(values, factors) #array of 6 old values for R2X

    new_R2X = percent_reduction_by_ligand(values, factors) #array of 5 by n_comp for R2X for each ligand after removing each component.

    percent_reduction = np.zeros_like(new_R2X)
    for ii in range(5):
        percent_reduction[ii,:] = 1 - new_R2X[ii, :] / old_R2X[ii]

    labels = ['Combined IL2 and IL15', 'IL7', 'IL9', 'IL4', 'IL21']
    colorsMarker = ['bo', 'ro', 'ko', 'mo', 'yo']
    for kk in range(5):
        ax.plot(range(1,factors[0].shape[1]+1), percent_reduction[kk,:], colorsMarker[kk], label = labels[kk])
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Percent Reduction in R2X')
    ax.legend()

def PCA_receptor(ax1, ax2, cell_names, data):
    """Plot PCA scores and loadings for Receptor Expression Data. """
    pca = PCA(n_components = 2)
    data = stats.zscore(data.astype(float), axis = 0)
    scores = pca.fit(data.T).transform(data.T) #34 cells by n_comp
    loadings = pca.components_ #n_comp by 8 receptors
    expVar = pca.explained_variance_ratio_

    colors = cm.rainbow(np.linspace(0, 1, 34))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '^', '4', 'P', '*', 'D', 's', 'X' ,'o', 'd', '1', '2', '3', '4', 'h', 'H', 'X', 'v', '*', '+', '8', 'P', 'p', 'D', '_','D', 's', 'X', 'o']
    markersReceptors = ['^', '4', 'P', '*', 'D', 's', 'X' ,'o']
    labelReceptors = ['IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL4Ra', 'IL21Ra']

    for ii in range(scores.shape[0]):
        ax1.scatter(scores[ii,0], scores[ii,1], c = colors[ii], marker = markersCells[ii], label = cell_names[ii])

    for jj in range(loadings.shape[1]):
        ax2.scatter(loadings[0,jj], loadings[1,jj], marker = markersReceptors[jj], label = labelReceptors[jj])

    x_max1 = np.max(np.absolute(np.asarray(ax1.get_xlim())))*1.1
    y_max1 = np.max(np.absolute(np.asarray(ax1.get_ylim())))*1.1

    x_max2 = np.max(np.absolute(np.asarray(ax2.get_xlim())))*1.1
    y_max2 = np.max(np.absolute(np.asarray(ax2.get_ylim())))*1.1

    ax1.set_xlim(-x_max1, x_max1)
    ax1.set_ylim(-y_max1, y_max1)
    ax1.set_xlabel('PC1 (' + str(round(expVar[0]*100, 2))+ '%)')
    ax1.set_ylabel('PC2 (' + str(round(expVar[1]*100, 2))+ '%)')
    ax1.set_title('Scores')
    ax1.legend(loc='upper left', bbox_to_anchor=(3.5, 1.735))

    ax2.set_xlim(-x_max2, x_max2)
    ax2.set_ylim(-y_max2, y_max2)
    ax2.set_xlabel('PC1 (' + str(round(expVar[0]*100, 2))+ '%)')
    ax2.set_ylabel('PC2 (' + str(round(expVar[1]*100, 2))+ '%)')
    ax2.set_title('Loadings')
    ax2.legend()

def plot_R2X(ax, tensor, factors_list, n_comps):
    """Function to plot R2X bar graph."""
    R2X_array = list()
    for n in range(n_comps):
        factors = factors_list[n]
        R2X = find_R2X(tensor, factors)
        R2X_array.append(R2X)
    ax.plot(range(1,n_comps+1), R2X_array, 'ko', label = 'Overall R2X')
    ax.set_ylabel('R2X')
    ax.set_xlabel('Number of Components')
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, n_comps+1))
    ax.set_xticklabels(np.arange(1, n_comps+1))
