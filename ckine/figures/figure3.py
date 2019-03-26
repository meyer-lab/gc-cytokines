"""
This creates Figure 3.
"""
import string
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints, values, mat, set_bounds
from ..Tensor_analysis import find_R2X, scale_all, perform_decomposition
from ..tensor_generation import data, cell_names

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 3, 4
    ax, f = getSetup((12, 9), (x, y))
    # Blank out for the cartoon
    ax[0].axis('off')
    ax[1].axis('off')
    ax[7].axis('off')
    ax[8].axis('off')
    ax[11].axis('off')

    factors_activity = []
    for jj in range(len(mat) - 1):
        factors = perform_decomposition(values , jj+1)
        factors_activity.append(factors)

    numpy_data = data.values[:,1:] # returns data values in a numpy array
    #['Il2ra' 'Il2rb' 'Il2rg' 'Il15ra'] in that order from Receptor levels. CD25, CD122, CD132, CD215

    n_comps = 4
    factors_activ = factors_activity[n_comps-1]
    newfactors_activ = factors_activ
    newfactors = scale_all(newfactors_activ)

    bar_receptors(ax[2], data)
    plot_R2X(ax[3], values, factors_activity, n_comps=5)

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    plot_timepoints(ax[4], newfactors[0]) #Change final input value depending on need

    for row in range(1, 3):
        subplotLabel(ax[row], string.ascii_uppercase[row]) # Add subplot labels
        compNum = 2*(row-1) + 1
        plot_cells(ax[row*y + 1], newfactors[1], compNum, compNum+1, cell_names, ax_pos = row*y + 1)
        plot_ligands(ax[row*y + 2], newfactors[2], compNum, compNum+1, ax_pos = row*y + 2)

        # Add labels and bounds
        set_bounds(row, y, ax, compNum)

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
