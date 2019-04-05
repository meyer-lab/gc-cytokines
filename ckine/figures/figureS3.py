"""
This creates Figure S3, which covers the Tucker factorization form.
"""
import string
import tensorly as tl
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints, values, set_bounds
from ..Tensor_analysis import perform_tucker, find_R2X_tucker
from ..tensor_generation import cell_names


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    x, y = 2, 3
    ax, f = getSetup((7.5, 5), (x, y), empts=[3])

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])  # Add subplot labels

    rank_list = [3, 2 * x, 2 * x]
    out = perform_tucker(values, rank_list)
    print(find_R2X_tucker(values, out))

    factors = out[1]
    plot_timepoints(ax[0], tl.to_numpy(factors[0]))

    for row in range(x):
        compNum = 2 * row + 1

        plot_cells(ax[row * y + 1], tl.to_numpy(factors[1]), compNum, compNum + 1, cell_names, ax_pos=row * y + 1, fig3=False)
        if compNum < rank_list[2]:
            plot_ligands(ax[row * y + 2], tl.to_numpy(factors[2]), compNum, compNum + 1, ax_pos=row * y + 2, fig3=False)

    f.tight_layout()

    return f
