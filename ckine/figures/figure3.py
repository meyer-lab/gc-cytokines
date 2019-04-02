"""
This creates Figure 3.
"""
import string
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints, values, mat, set_bounds
from ..Tensor_analysis import find_R2X, perform_decomposition
from ..tensor_generation import data, cell_names

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 3, 4
    ax, f = getSetup((12, 9), (x, y), mults=[0], multz={0:2}, empts=[7,8,11])
    # Blank out for the cartoon
    ax[0].axis('off')

    factors_activity = []
    for jj in range(len(mat) - 1):
        factors = perform_decomposition(values , jj+1)
        factors_activity.append(factors)

    numpy_data = data.values[:,1:] # returns data values in a numpy array
    #['Il2ra' 'Il2rb' 'Il2rg' 'Il15ra'] in that order from Receptor levels. CD25, CD122, CD132, CD215

    n_comps = 4
    factors_activ = factors_activity[n_comps-1]

    bar_receptors(ax[1], data)
    plot_R2X(ax[2], values, factors_activity, n_comps=5)

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii]) # Add subplot labels

    plot_timepoints(ax[3], factors_activ[0]) #Change final input value depending on need

    for row in range(1, 3):
        compNum = 2*(row-1) + 1
        plot_cells(ax[row*2 + 2], factors_activ[1], compNum, compNum+1, cell_names, ax_pos = row*2 + 2)
        plot_ligands(ax[row*2 +3], factors_activ[2], compNum, compNum+1, ax_pos = row*2 +3)

    f.tight_layout()

    return f

def bar_receptors(ax, data):
    """Plot Bar graph for Receptor Expression Data. """
    data.plot.bar(x = "Cell Type", rot = 30, ax = ax)
    ax.legend(loc = 1)
    ax.set_ylabel("Surface Receptor [# / cell]")

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
