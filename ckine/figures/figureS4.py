"""
This creates Figure S4, which covers the Tucker factorization form.
"""
import string
import logging
import numpy as np
import seaborn as sns
import tensorly as tl
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints
from .figure3 import values
from ..tensor import perform_tucker, find_R2X_tucker
from ..make_tensor import tensor_time
from ..imports import import_Rexpr

cell_dim = 1  # For this figure, the cell dimension is along the second [python index 1].


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    x, y = 2, 3
    ax, f = getSetup((7.5, 4.5), (x, y))

    _, _, cell_names = import_Rexpr()
    for ii, item in enumerate(ax[0:5]):
        subplotLabel(item, string.ascii_uppercase[ii])  # Add subplot labels

    rank_list = [2, 2, 3]
    out = perform_tucker(values, rank_list)
    RtwoX = find_R2X_tucker(values, out)
    logging.info(RtwoX)
    assert RtwoX > 0.9
    logging.info(out[0].shape)

    plot_core(ax[3], out[0][0, :, :])
    ax[3].set_title("Time Component 1")
    plot_core(ax[4], out[0][1, :, :])
    ax[4].set_title("Time Component 2")
    factors = out[1]
    plot_timepoints(ax[0], tensor_time, tl.to_numpy(factors[0]))
    plot_cells(ax[1], tl.to_numpy(factors[1]), 1, 2, cell_names)
    plot_ligands(ax[2], tl.to_numpy(factors[2]), ligand_names=['IL-2', 'IL-15', 'IL-7'])

    # Blank out
    ax[5].axis("off")

    return f


def plot_core(ax, core):
    """Generate heatmaps for the core tensor."""
    # Begin by unfolding the core tensor on its 3 faces.
    sns.heatmap(np.squeeze(core), cmap="YlGnBu", cbar=True, ax=ax, annot=True, square=True, fmt='.1f', xticklabels=['1', '2', '3'], yticklabels=['1', '2'])
    ax.set_xlabel("Ligands")
    ax.set_ylabel("Cells")
