"""
This creates Figure 5. CP decomposition of measured pSTAT data.
"""
import string
import numpy as np
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints
from .figure3 import plot_R2X
from ..tensor import perform_decomposition, z_score_values
from ..imports import import_pstat

cell_dim = 0  # For this figure, the cell dimension is along the first [python index 0].


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 2.5), (1, 4))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])  # Add subplot labels

    _, cell_names, IL2_data, IL15_data, _, _ = import_pstat()

    IL2 = np.flip(IL2_data, axis=(0, 1))  # Makes them in both chronological order and ascending stimulation concentration
    IL15 = np.flip(IL15_data, axis=(0, 1))  # Makes them in both chronological order and ascending stimulation concentration
    IL2 = np.insert(IL2, range(0, IL2.shape[0], 4), 0.0, axis=0)  # add in a zero value for the activity at t=0
    IL15 = np.insert(IL15, range(0, IL15.shape[0], 4), 0.0, axis=0)  # add in a zero value for the activity at t=0
    concat = np.concatenate((IL2, IL15), axis=1)  # Prepare for tensor reshaping
    measured_tensor = np.reshape(concat, (len(cell_names), 5, IL2.shape[1] * 2))
    measured_tensor = z_score_values(measured_tensor, cell_dim)

    factors_activity = []
    for jj in range(5):
        factors = perform_decomposition(measured_tensor, jj + 1)
        factors_activity.append(factors)

    plot_R2X(ax[0], measured_tensor, factors_activity)

    n_comps = 2
    factors_activ = factors_activity[n_comps - 1]  # First dimension is cells. Second is time. Third is ligand.
    plot_timepoints(ax[1], np.array([0.0, 0.5, 1., 2., 4.]) * 60., factors_activ[1])  # Time is the second dimension in this case because reshaping only correctly did 11*4*24
    plot_cells(ax[2], factors_activ[0], 1, 2, cell_names)
    plot_ligands(ax[3], factors_activ[2], ligand_names=['IL-2', 'IL-15'])

    return f


def correlation_cells(ax, experimental, predicted, cell_names):
    """Function that takes in predicted and experimental components from cell decomposion and plots them against each other."""
    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '4', 'H', 'P', '*', 'D', 's', 'X']
    for ii, cell_name in enumerate(cell_names):
        ax.scatter(experimental, predicted, c=[colors[ii]], marker=markersCells[ii], label=cell_name, alpha=0.75)


def correlation_ligands(ax, experimental, predicted):
    """Function that takes in predicted and experimental components from ligand decomposion and plots them against each other."""
    ax.scatter(experimental, predicted)
