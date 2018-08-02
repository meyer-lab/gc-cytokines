"""
This creates Figure 3.
"""
import string
from .figureCommon import subplotLabel, getSetup
import numpy as np
import os
import pickle
from ..tensor_generation import prepare_tensor
from ..Tensor_analysis import find_R2X, split_R2X, R2X_singles


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

    with open(factors_filename,'rb') as ff:
        factors_list = pickle.load(ff)
    factors = factors_list[19]

    values, _, _, _, _ = prepare_tensor(2)
    n_comps = 20

    plot_R2X(ax[3], values, factors_list, n_comps)
    
    plot_split_R2X(ax[3], values, factors_list, n_comps)
    
    plot_R2X_singles(ax[7], values, factors_list, n_comps)
    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    f.tight_layout()

    return f

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
    ax.set_xticks(np.arange(1, n_comps+1, 2))
    ax.set_xticklabels(np.arange(1, n_comps+1, 2))
    ax.legend()

def plot_R2X_singles(ax, values, factors_list, n_comps):
    """R2X plot for removing single components from final factorization."""
    R2X_matrix = split_R2X(values, factors_list, n_comps)
    R2X_singles_matrix = R2X_singles(values, factors_list, n_comps)
    old_R2X = np.zeros((3,n_comps))

    for ii in range(3):
        old_R2X[ii,:] = R2X_matrix[ii,-1] #the -1 value here is for the R2X for the last component. 

    percent_reduction = 1 - R2X_singles_matrix[1:4, :] / old_R2X

    ax.plot(range(1,n_comps+1), percent_reduction[0,:], 'bo', label = 'Ligand Activity R2X')
    ax.plot(range(1,n_comps+1), percent_reduction[1,:], 'ro', label = 'Surface Receptors R2X')
    ax.plot(range(1,n_comps+1), percent_reduction[2,:], 'go', label = 'Total Receptors R2X')    
    ax.set_ylabel('Percent Reduction in R2X')
    ax.set_xlabel('Component Index')
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, n_comps+1, 2))
    ax.set_xticklabels(np.arange(1, n_comps+1, 2))
    ax.legend()

def plot_split_R2X(ax, tensor, factors_list, n_comps):
    """This function takes in the values tensor, splits it up into a mini tensor corresponding to quantity type."""
    R2X_matrix = split_R2X(tensor, factors_list, n_comps)
    ax.plot(range(1,n_comps+1), R2X_matrix[0,:], 'bo', label = 'Ligand Activity R2X')
    ax.plot(range(1,n_comps+1), R2X_matrix[1,:], 'ro', label = 'Surface Receptors R2X')
    ax.plot(range(1,n_comps+1), R2X_matrix[2,:], 'go', label = 'Total Receptors R2X')
    ax.legend()
