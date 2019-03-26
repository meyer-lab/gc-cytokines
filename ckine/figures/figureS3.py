"""
This creates Figure S2, which covers the Tucker factorization form.
"""
import string
import numpy as np
import tensorly as tl
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints, values, set_bounds
from ..Tensor_analysis import perform_tucker, find_R2X_tucker
from ..tensor_generation import data, cell_names

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    x, y = 2, 3
    ax, f = getSetup((7.5, 5), (x, y))
    ax[3].axis('off')

    rank_list = [3, 2*x, 2*x]
    out = perform_tucker(values, rank_list)
    print(find_R2X_tucker(values, out))

    factors = out[1]
    plot_timepoints(ax[0], tl.to_numpy(factors[0]))

    for row in range(x):
        subplotLabel(ax[row], string.ascii_uppercase[row]) # Add subplot labels
        compNum = 2*row + 1

        plot_cells(ax[row*y + 1], tl.to_numpy(factors[1]), compNum, compNum + 1, cell_names, ax_pos = row*y + 1)
        if compNum < rank_list[2]:
            plot_ligands(ax[row*y + 2], tl.to_numpy(factors[2]), compNum, compNum + 1, ax_pos = row*y + 2, fig3 = False)

        # Add labels and bounds
        set_bounds(row, y, ax, compNum)


    subplotLabel(ax[3], string.ascii_uppercase[2])        
    f.tight_layout()

    return f
