"""
This creates Figure S2.
"""
import string
import numpy as np
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup
from ..tensor_generation import prepare_tensor
from ..Tensor_analysis import perform_decomposition


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 5, 5
    ax, f = getSetup((10, 10), (x, y))

    values, _, _, _, cell_names = prepare_tensor(2)
    factors = perform_decomposition(values, 2*x)

    for row in range(x):
        compNum = 2*row + 1
        plot_timepoint(ax[row*y], factors[0], compNum, compNum+1)
        plot_cells(ax[row*y + 1], factors[1], compNum, compNum+1, cell_names)
        plot_ligands(ax[row*y + 2], factors[2], compNum, compNum+1)
        plot_values(ax[row*y + 3], factors[3], compNum, compNum+1)

        # Set axes to center on the origin, and add labels
        for col in range(y):
            ax[row*y + col].set_xlabel('Component ' + str(compNum))
            ax[row*y + col].set_ylabel('Component ' + str(compNum+1))

            x_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_xlim())))
            y_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_ylim())))

            ax[row*y + col].set_xlim(-x_max, x_max)
            ax[row*y + col].set_ylim(-y_max, y_max)

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    f.tight_layout()

    return f


def plot_values(ax, factors, component_x, component_y):
    """Plot the values decomposition factors matrix."""
    #Generate a plot for component x vs component y of the factors[3] above representing our values
    labels = ['IL2', 'IL15', 'IL7', 'IL9', 'IL4','IL21','IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL4Ra','IL21Ra','IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL4Ra','IL21Ra']
    #Set Active to color red. Set Surface to color blue. Set Total to color black
    ax.scatter(factors[0:6, component_x - 1], factors[0:6, component_y - 1], color = 'r', label = 'Ligand Activity')
    ax.scatter(factors[6:14, component_x - 1], factors[6:14, component_y - 1], color = 'b', label = 'Surface Receptor')
    ax.scatter(factors[14::, component_x - 1], factors[14::, component_y - 1], color = 'k', label = 'Total Receptor')

    for i, item in enumerate(labels):
        ax.annotate(item, xy=(factors[i, component_x - 1], factors[i, component_y - 1]), xytext = (0, 0), textcoords = 'offset points')


def plot_timepoint(ax, factors, component_x, component_y):
    """Plot the timepoint decomposition in the first column of figS2."""
    ax.scatter(factors[:, component_x - 1], factors[:, component_y - 1], color = 'k')
    ax.annotate(str(1000), xy=(factors[999, component_x - 1], factors[999, component_y - 1]), xytext = (0, 0), textcoords = 'offset points')


def plot_cells(ax, factors, component_x, component_y, cell_names):
    """This function plots the combination decomposition based on cell type."""
    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    ax.scatter(factors[:, component_x - 1], factors[:, component_y - 1], c=colors, label = cell_names)


def plot_ligands(ax, factors, component_x, component_y):
    "This function is to plot the ligand combination dimension of the values tensor."
    ax.scatter(factors[:,component_x - 1], factors[:,component_y - 1])
