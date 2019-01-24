"""
This creates Figure S2.
"""
import string
import os
import pickle
import numpy as np, cupy as cp
import pandas as pds
import tensorly
from tensorly.decomposition import tucker
tensorly.set_backend('cupy')
from .figureCommon import subplotLabel, getSetup, plot_timepoint, plot_cells, plot_ligands, plot_values, plot_timepoints
from ..Tensor_analysis import perform_tucker, find_R2X_tucker

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    x, y = 5, 4
    ssize = 3
    ax, f = getSetup((ssize*y, ssize*x), (x, y))

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    expr_filename = os.path.join(fileDir, './ckine/data/expr_table.csv')
    data = pds.read_csv(expr_filename) # Every column in the data represents a specific cell
    cell_names = data.columns.values.tolist()[1::] #returns the cell names from the pandas dataframe (which came from csv)


    factors_filename = os.path.join(fileDir, './ckine/data/factors_results/Sampling.pickle')
    factors_filename = os.path.abspath(os.path.realpath(factors_filename))

    with open(factors_filename,'rb') as ff:
        two_files = pickle.load(ff)
    
    values = cp.asnumpy(tensorly.tucker_to_tensor(two_files[1][0], two_files[1][1])) #This reconstructs our values tensor from the decomposed one that we used to store our data in.
    values = values[:,:,:,[0,1,2,3,4]]
    rank_list = [2,10,8,5]
    out = perform_tucker(values, rank_list)

    factors = out[1]
    plot_timepoints(ax[0], cp.asnumpy(factors[0]))

    for row in range(x):
        subplotLabel(ax[row], string.ascii_uppercase[row]) # Add subplot labels
        compNum = 2*row + 1
        if row >= 1:
            ax[row*y].axis('off')

        if row >= np.floor(rank_list[2]/2):
            ax[row*y + 2].axis('off')

        if row > np.floor(rank_list[3]/2):
            ax[row*y +3].axis('off')
        
        plot_cells(ax[row*y + 1], cp.asnumpy(factors[1]), compNum, compNum+1, cell_names, ax_pos = row*y + 1)
        if compNum < rank_list[2]:
            plot_ligands(ax[row*y + 2], cp.asnumpy(factors[2]), compNum, compNum+1)
        if compNum < rank_list[3]:
            plot_values(ax[row*y + 3] , cp.asnumpy(factors[3]), compNum, compNum+1, ax_pos = row*y + 3)
        elif compNum == rank_list[3]:
            plot_values(ax[row*y + 3] , cp.asnumpy(factors[3]), compNum-1, compNum, ax_pos = row*y + 3)

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
    print('Finding R2X')
    print('Finding R2X')
    print(find_R2X_tucker(values, out, subt = True))
    return f
