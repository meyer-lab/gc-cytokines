"""
This creates Figure 3.
"""
import string
from .figureCommon import subplotLabel, getSetup
import numpy as np
from ..tensor_generation import prepare_tensor
from ..Tensor_analysis import perform_decomposition, find_R2X


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 3, 4
    ax, f = getSetup((10, 10), (x, y))

    # Blank out for the cartoon
    ax[0].axis('off')

    values, _, _, _, cell_names = prepare_tensor(2,50)
    plot_R2X(ax[3], values, n_comps = 17)

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
    ax.set_xticks(np.arange(1, n_comps+1))
    ax.set_xticklabels(np.arange(1, n_comps+1))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('R2X')
