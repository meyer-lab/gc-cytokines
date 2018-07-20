"""
This creates Figure S2.
"""
import string
import numpy as np
import itertools
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup
from ..tensor_generation import prepare_tensor
from ..Tensor_analysis import perform_decomposition

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 10, 4
    ssize = 3
    ax, f = getSetup((ssize*y, ssize*x), (x, y))

    values, _, _, _, cell_names = prepare_tensor(2)
    factors = perform_decomposition(values, 2*x)

    for row in range(x):
        subplotLabel(ax[row*y], string.ascii_uppercase[row]) # Add subplot labels

        compNum = 2*row + 1
        plot_timepoint(ax[row*y], factors[0], compNum, compNum+1)
        plot_cells(ax[row*y + 1], factors[1], compNum, compNum+1, cell_names)
        plot_ligands(ax[row*y + 2], factors[2], compNum, compNum+1)
        plot_values(ax[row*y + 3], factors[3], compNum, compNum+1)

        # Set axes to center on the origin, and add labels
        for col in range(y):
            ax[row*y + col].set_xlabel('Component ' + str(compNum))
            ax[row*y + col].set_ylabel('Component ' + str(compNum+1))

            x_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_xlim())))*1.1
            y_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_ylim())))*1.1

            ax[row*y + col].set_xlim(-x_max, x_max)
            ax[row*y + col].set_ylim(-y_max, y_max)

    f.tight_layout()

    return f


def plot_values(ax, factors, component_x, component_y):
    """Plot the values decomposition factors matrix."""
    #Generate a plot for component x vs component y of the factors[3] above representing our values
    # The markers are for the following elements in order: 'IL2', 'IL15', 'IL7', 'IL9', 'IL4','IL21','IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL4Ra','IL21Ra','IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL4Ra','IL21Ra.'
    
    markersLigand = itertools.cycle(('^', '*', 'D', 's', 'X', 'o'))
    markersReceptors = itertools.cycle(('^', '4', 'P', '*', 'D', 's', 'X' ,'o')) 
    
    #Set Active to color red. Set Surface to color blue. Set Total to color black
    for q,p in zip(factors[0:6, component_x - 1], factors[0:6, component_y - 1]):
        ax.plot(q, p, linestyle = '', c = 'r', marker = next(markersLigand), label = 'Ligand Activity')

    for q,p in zip(factors[6:14, component_x - 1], factors[6:14, component_y - 1]):
        ax.plot(q, p, linestyle = '', c = 'b', marker = next(markersReceptors), label = 'Surface Receptor')
    
    for q,p in zip(factors[14::, component_x - 1], factors[14::, component_y - 1]):
        ax.plot(q, p, linestyle = '', c = 'k', marker = next(markersReceptors), label = 'Total Receptor')


def plot_timepoint(ax, factors, component_x, component_y):
    """Plot the timepoint decomposition in the first column of figS2."""
    ax.plot(factors[:, component_x - 1], factors[:, component_y - 1], color = 'k')
    ax.scatter(factors[-1, component_x - 1], factors[-1, component_y - 1], s = 12, color = 'b')

def plot_cells(ax, factors, component_x, component_y, cell_names):
    """This function plots the combination decomposition based on cell type."""
    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    ax.scatter(factors[:, component_x - 1], factors[:, component_y - 1], c=colors, label = cell_names)

def plot_ligands(ax, factors, component_x, component_y):
    "This function is to plot the ligand combination dimension of the values tensor."
    ax.scatter(factors[:,component_x - 1], factors[:,component_y - 1])
