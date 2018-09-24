"""
This creates Figure S2.
"""
import string
import numpy as np
import pandas as pds
import os
import pickle
from ..tensor_generation import prepare_tensor
from .figureCommon import subplotLabel, getSetup, plot_timepoint, plot_cells, plot_ligands, plot_values
from ..Tensor_analysis import reorient_factors, perform_tucker
from .figureCommon import subplotLabel, getSetup, plot_timepoint, plot_cells, plot_ligands, plot_values, plot_timepoints
from ..Tensor_analysis import reorient_factors, find_R2X_tucker

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    x, y = 5, 4
    ssize = 3
    ax, f = getSetup((ssize*y, ssize*x), (x, y))
    
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    expr_filename = os.path.join(fileDir, './ckine/data/expr_table.csv')
    data = pds.read_csv(expr_filename) # Every column in the data represents a specific cell
    cell_names = data.columns.values.tolist()[1::] #returns the cell names from the pandas dataframe (which came from csv)
    
    values, _, _, _, _ = prepare_tensor(2)
    values = values[:,:,:,[0,1,2,3,4]]
    rank_list = [2,10,8,5]
    out = perform_tucker(values, rank_list)

    factors = out[1]
    plot_timepoints(ax[0], factors[0])

    for row in range(x):
        subplotLabel(ax[row], string.ascii_uppercase[row]) # Add subplot labels
        compNum = 2*row + 1
        if row >= 1:
            ax[row*y].axis('off')

        if row >= np.floor(rank_list[2]/2):
            ax[row*y + 2].axis('off')

        if row > np.floor(rank_list[3]/2):
            ax[row*y +3].axis('off')
        
        plot_cells(ax[row*y + 1], factors[1], compNum, compNum+1, cell_names, ax_pos = row*y + 1)
        if compNum < rank_list[2]:
            plot_ligands(ax[row*y + 2], factors[2], compNum, compNum+1)
        if compNum < rank_list[3]:
            plot_values(ax[row*y + 3] , factors[3], compNum, compNum+1, ax_pos = row*y + 3)
        elif compNum == rank_list[3]:
            plot_values(ax[row*y + 3] , factors[3], compNum-1, compNum, ax_pos = row*y + 3)


        # Set axes to center on the origin, and add labels
        for col in range(1,y):
            ax[row*y + col].set_xlabel('Component ' + str(compNum))
            ax[row*y + col].set_ylabel('Component ' + str(compNum+1))
            
            if compNum == rank_list[3] and col == 3:
                ax[row*y + col].set_xlabel('Component ' + str(compNum-1))
                ax[row*y + col].set_ylabel('Component ' + str(compNum))

            x_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_xlim())))*1.1
            y_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_ylim())))*1.1

            ax[row*y + col].set_xlim(-x_max, x_max)
            ax[row*y + col].set_ylim(-y_max, y_max)

    print(find_R2X_tucker(values, out, subt = True))
    return f
