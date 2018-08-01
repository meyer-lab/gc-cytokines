"""
This creates Figure S2.
"""
import string
import numpy as np
import pandas as pds
import os
import pickle
import itertools
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from .figureCommon import subplotLabel, getSetup

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    factors_filename = os.path.join(fileDir, './ckine/data/factors_results/Sampling.pickle')
    factors_filename = os.path.abspath(os.path.realpath(factors_filename))

    expr_filename = os.path.join(fileDir, './ckine/data/expr_table.csv')
    data = pds.read_csv(expr_filename) # Every column in the data represents a specific cell
    cell_names = data.columns.values.tolist()[1::] #returns the cell names from the pandas dataframe (which came from csv)

    with open(factors_filename,'rb') as f:
        factors_list = pickle.load(f)
    factors = factors_list[19]

    x, y = 10, 4
    ssize = 3
    ax, f = getSetup((ssize*y, ssize*x), (x, y))
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
    #Set Active to color red. Set Surface to color blue. Set Total to color black
    markersLigand = itertools.cycle(('^', '*', 'D', 's', 'X', 'o'))
    markersReceptors = itertools.cycle(('^', '4', 'P', '*', 'D', 's', 'X' ,'o'))
    
    for q,p in zip(factors[0:6, component_x - 1], factors[0:6, component_y - 1]):
        ax.plot(q, p, linestyle = '', c = 'r', marker = next(markersLigand), label = 'Ligand Activity')
        
    for q,p in zip(factors[6:14, component_x - 1], factors[6:14, component_y - 1]):
        ax.plot(q, p, linestyle = '', c = 'b', marker = next(markersReceptors), label = 'Surface Receptor')
        
    for q,p in zip(factors[14::, component_x - 1], factors[14::, component_y - 1]):
        ax.plot(q, p, linestyle = '', c = 'k', marker = next(markersReceptors), label = 'Total Receptor')

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_timepoint(ax, factors, component_x, component_y):
    """Plot the timepoint decomposition in the first column of figS2."""
    ax.plot(factors[:, component_x - 1], factors[:, component_y - 1], color = 'k')
    ax.scatter(factors[-1, component_x - 1], factors[-1, component_y - 1], s = 12, color = 'b')

def plot_cells(ax, factors, component_x, component_y, cell_names):
    """This function plots the combination decomposition based on cell type."""
    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '^', '4', 'P', '*', 'D', 's', 'X' ,'o', 'd', '1', '2', '3', '4', 'h', 'H', 'X', 'v', '*', '+', '8', 'P', 'p', 'D', '_','D', 's', 'X', 'o']

    for ii in range(len(factors[:, component_x - 1])):
        ax.scatter(factors[ii, component_x - 1], factors[ii, component_y - 1], c = colors[ii], marker = markersCells[ii], label = cell_names[ii])
    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_ligands(ax, factors, component_x, component_y):
    "This function is to plot the ligand combination dimension of the values tensor."
    ax.scatter(factors[:,component_x - 1], factors[:,component_y - 1])
