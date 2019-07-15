"""
This creates Figure 5.
"""
import string
import tensorly as tl
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_timepoints, plot_R2X, plot_ligands
from ..imports import import_Rexpr
from ..tensor import perform_decomposition, z_score_values
from ..make_tensor import make_tensor, tensor_time

cell_dim = 1  # For this figure, the cell dimension is along the second [python index 1].
values, _, _, _, _ = make_tensor(mut=True)
values = z_score_values(tl.tensor(values), cell_dim)


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 2.5), (1, 4))

    _, _, cell_names = import_Rexpr()
    factors_activity = []
    for jj in range(5):
        factors = perform_decomposition(values, jj + 1)
        factors_activity.append(factors)

    n_comps = 2
    factors_activ = factors_activity[n_comps - 1]

    plot_R2X(ax[0], values, factors_activity)

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])  # Add subplot labels

    plot_timepoints(ax[1], tensor_time, factors_activ[0])

    plot_cells(ax[2], factors_activ[1], 1, 2, cell_names)

    plot_ligands(ax[3], factors_activ[2], ligand_names=['IL-2', 'IL-2Ra mut', 'IL-2Rb mut'])

    return f
