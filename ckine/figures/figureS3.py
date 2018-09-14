"""
This creates Figure S1.
"""
import string
import numpy as np
import pandas as pds
import os
import pickle
from ..tensor_generation import prepare_tensor
from .figureCommon import subplotLabel, getSetup, plot_timepoint, plot_cells, plot_ligands, plot_values
from ..Tensor_analysis import reorient_factors
from .figureCommon import subplotLabel, getSetup, plot_timepoint, plot_cells, plot_ligands, plot_values
from ..Tensor_analysis import reorient_factors, find_R2X


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
        factors_activity = pickle.load(f)[1]

    n_comps = 5
    factors = factors_activity[n_comps]
    newfactors = reorient_factors(factors)

    x, y = 3, 4
    ssize = 3
    ax, f = getSetup((ssize*y, ssize*x), (x, y))

    plot_timepoints(ax[0], newfactors[0])

    
    for row in range(x):
        subplotLabel(ax[row], string.ascii_uppercase[row]) # Add subplot labels
        compNum = 2*row + 1
        if row >= 1:
            ax[row*y].axis('off')
        plot_cells(ax[row*y + 1], newfactors[1], compNum, compNum+1, cell_names, ax_pos = row*y + 1)
        plot_ligands(ax[row*y + 2], newfactors[2], compNum, compNum+1)
        plot_values(ax[row*y + 3] ,newfactors[3], compNum, compNum+1, ax_pos = row*y + 3)

        # Set axes to center on the origin, and add labels
        for col in range(1,y):
            ax[row*y + col].set_xlabel('Component ' + str(compNum))
            ax[row*y + col].set_ylabel('Component ' + str(compNum+1))

            x_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_xlim())))*1.1
            y_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_ylim())))*1.1

            ax[row*y + col].set_xlim(-x_max, x_max)
            ax[row*y + col].set_ylim(-y_max, y_max)

    values, _, _, _, _ = prepare_tensor(2)
    for n in range(6):
        factors = factors_activity[n]
        R2X = find_R2X(np.concatenate((values[:,:,:,[0,1,2,3,4]], values[:,:,:,[0,1,2,3,4]]), axis = 3), factors, subt = False)
        print(R2X)
    return f

def plot_timepoints(ax, factors):
    """Function to put all timepoint plots in one figure."""
    ts = np.logspace(-3., np.log10(4 * 60.), 100)
    ts = np.insert(ts, 0, 0.0)
    colors = ['b', 'k', 'r', 'y', 'm', 'g']
    for ii in range(factors.shape[1]):
        ax.plot(ts, factors[:,ii], c = colors[ii], label = 'Component ' + str(ii+1))
        ax.scatter(ts[-1], factors[-1, ii], s = 12, color = 'k')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Component')
    ax.legend()
