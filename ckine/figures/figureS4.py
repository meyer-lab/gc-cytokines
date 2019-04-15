"""
This creates Figure S4, which covers the Tucker factorization form.
"""
import string
import seaborn  as sns
import tensorly as tl
from tensorly import unfold
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints, values, set_bounds
from ..Tensor_analysis import perform_tucker, find_R2X_tucker
from ..tensor_generation import cell_names

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    x, y = 2, 3
    ax, f = getSetup((7.5, 5), (x, y), empts=[3])

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])  # Add subplot labels

    rank_list = [3, 3, 4]
    out = perform_tucker(values, rank_list)
    print(out[0])
    print(find_R2X_tucker(values, out))

    plot_core(ax[3], out[0])
    factors = out[1]
    plot_timepoints(ax[0], tl.to_numpy(factors[0]))
    plot_cells(ax[1], tl.to_numpy(factors[1]), 1, 2, cell_names, ax_pos= 1, fig3=False)
    plot_cells(ax[4], tl.to_numpy(factors[1]), 2, 3, cell_names, ax_pos= 4, fig3=False)

    for row in range(2):
        compNum = 2 * row + 1
        plot_ligands(ax[row * y + 2], tl.to_numpy(factors[2]), compNum, compNum + 1, ax_pos= row * y + 2, fig3=False)

    f.tight_layout()

    return f

def plot_core(ax, core):
    """Generate heatmaps for the core tensor."""
    #Begin by unfolding the core tensor on its 3 faces.
    X3 = unfold(core, 2)
    sns.heatmap(X3,cmap="YlGnBu",cbar=True,ax=ax)
