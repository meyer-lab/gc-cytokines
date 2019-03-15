"""
This creates Figure S2.
"""
import string
import os
import pickle
import numpy as np
import pandas as pds
import tensorly as tl
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints, n_ligands, values
from ..Tensor_analysis import perform_tucker, find_R2X_tucker
from ..tensor_generation import data

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    x, y = 3, 3
    ssize = 4
    ax, f = getSetup((ssize*y, ssize*x), (x, y))

    numpy_data = data.values[:,1:] # returns data values in a numpy array
    cell_names = ['Naive Th', 'Mem Th', 'Naive Treg', 'Mem Treg','Naive CD8+', 'Mem CD8+','NK','NKT']
    #['Il2ra' 'Il2rb' 'Il2rg' 'Il15ra'] in that order from Receptor levels. CD25, CD122, CD132, CD215

    rank_list = [3, 6, 6]
    out = perform_tucker(values, rank_list)
    print(find_R2X_tucker(values, out))

    factors = out[1]
    plot_timepoints(ax[0], tl.to_numpy(factors[0]))

    for row in range(x):
        subplotLabel(ax[row], string.ascii_uppercase[row]) # Add subplot labels
        compNum = 2*row + 1
        if row >= 1:
            ax[row*y].axis('off')

        plot_cells(ax[row*y + 1], tl.to_numpy(factors[1]), compNum, compNum+1, cell_names, ax_pos = row*y + 1)
        if compNum < rank_list[2]:
            plot_ligands(ax[row*y + 2], tl.to_numpy(factors[2]), compNum, compNum+1, ax_pos = row*y + 2)

        # Set axes to center on the origin, and add labels
        for col in range(1,y):
            ax[row*y + col].set_xlabel('Component ' + str(compNum))
            ax[row*y + col].set_ylabel('Component ' + str(compNum+1))

            x_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_xlim())))*1.1
            y_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_ylim())))*1.1

            ax[row*y + col].set_xlim(-x_max, x_max)
            ax[row*y + col].set_ylim(-y_max, y_max)
    return f
