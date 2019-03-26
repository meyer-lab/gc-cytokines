"""
This creates Figure S2, which covers the Tucker factorization form.
"""
import string
import numpy as np
import tensorly as tl
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints, values
from ..Tensor_analysis import perform_tucker, find_R2X_tucker
from ..tensor_generation import data, cell_names

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    x, y = 2, 3
    ax, f = getSetup((9, 5), (x, y))
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

        # Set axes to center on the origin, and add labels
        for col in range(1, y):
            ax[row*y + col].set_xlabel('Component ' + str(compNum))
            ax[row*y + col].set_ylabel('Component ' + str(compNum+1))

            x_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_xlim())))*1.1
            y_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_ylim())))*1.1

            ax[row*y + col].set_xlim(-0.05, x_max)
            ax[row*y + col].set_ylim(-0.05, y_max)

    subplotLabel(ax[3], string.ascii_uppercase[2])        
    f.tight_layout()

    return f
