"""
This creates Figure 3.
"""
import string
from .figureCommon import subplotLabel, getSetup
import numpy as np
from ..tensor_generation import prepare_tensor
from ..Tensor_analysis import perform_decomposition, find_R2X, split_R2X


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 3, 4
    ax, f = getSetup((10, 10), (x, y))

    # Blank out for the cartoon
    ax[0].axis('off')

    values, _, _, _, cell_names = prepare_tensor(2,50)
    n_comps = 17

    plot_R2X(ax[3], values, n_comps)
    plot_split_R2X(ax[4], ax[5], ax[6], values, n_comps)
    
    for n in range(3,7):
        ax[n].set_xticks(np.arange(1, n_comps+1))
        ax[n].set_xticklabels(np.arange(1, n_comps+1))
        ax[n].set_ylim(0, 1)
        ax[n].set_xlabel('Number of Components')
    
    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    f.tight_layout()

    return f

def plot_R2X(ax, tensor, n_comps):
    """Function to plot R2X bar graph."""
    R2X_array = list()
    for n in range(1, n_comps + 1):
        R2X = find_R2X(tensor, n)
        R2X_array.append(R2X)
    ax.bar(range(1,n_comps+1), R2X_array)
    ax.set_ylabel('R2X')

def plot_split_R2X(ax1, ax2, ax3, tensor, n_comps):
    """This function takes in the values tensor, splits it up into a mini tensor corresponding to the splitType. If splitType =1, then split ligand activities, if =2, then split surface receptors, if = 3, then split total receptors."""
    R2X_matrix = split_R2X(tensor, n_comps)

    ax1.bar(range(1,n_comps+1), R2X_matrix[0,:])
    ax1.set_ylabel('Ligand Activity R2X')

    ax2.bar(range(1,n_comps+1), R2X_matrix[1,:])
    ax2.set_ylabel('Surface Receptors R2X')

    ax3.bar(range(1,n_comps+1), R2X_matrix[2,:])
    ax3.set_ylabel('Total Receptors R2X')