"""
This creates Figure 5.
"""
import string
import tensorly as tl
import numpy as np
from matplotlib.lines import Line2D
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_timepoints, plot_R2X
from ..imports import import_Rexpr, import_pstat
from ..tensor import perform_decomposition
from ..make_tensor import make_tensor, n_lig

cell_dim = 1  # For this figure, the cell dimension is along the second [python index 1].
values, _, mat, _, _ = make_tensor(mut=True)
values = tl.tensor(values)


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 2, 3
    ax, f = getSetup((7.5, 5), (x, y), empts=[5])

    n_ligands = n_lig(mut=True)
    _, _, cell_names = import_Rexpr()
    factors_activity = []
    for jj in range(len(mat) - 1):
        factors = perform_decomposition(values, jj + 1, cell_dim)
        factors_activity.append(factors)

    n_comps = 3
    factors_activ = factors_activity[n_comps - 1]

    plot_R2X(ax[0], values, factors_activity, n_comps=5, cells_dim=cell_dim)

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])  # Add subplot labels

    plot_timepoints(ax[1], factors_activ[0])  # Change final input value depending on need

    plot_cells(ax[2], factors_activ[1], 1, 2, cell_names, ax_pos=2)
    plot_cells(ax[3], factors_activ[1], 2, 3, cell_names, ax_pos=6)

    plot_ligands(ax[4], factors_activ[2], n_ligands=n_ligands, mesh=mat)

    return f

def plot_ligands(ax, factors, n_ligands, mesh):
    """Function to put all ligand decomposition plots in one figure."""
    ILs, _, _, _ = import_pstat()  # Cytokine stimulation concentrations in nM
    ILs = np.flip(ILs)
    colors = ['b', 'k', 'r', 'y', 'm', 'g']
    markers = ['^', '*', '.']
    legend_shape = [Line2D([0], [0], color='k', marker=markers[0], label='IL-2', linestyle=''),
                    Line2D([0], [0], color='k', label='IL-2Ra mut', marker=markers[1], linestyle=''),
                    Line2D([0], [0], color='k', label='IL-2Rb mut', marker=markers[2], linestyle='')]

    for ii in range(factors.shape[1]):
        for jj in range(n_ligands):
            idx = range(jj * int(mesh.shape[0] / n_ligands), (jj + 1) * int(mesh.shape[0] / n_ligands))
            if jj == 0:
                ax.plot(ILs, factors[idx, ii], color=colors[ii], label='Component ' + str(ii + 1))
                ax.scatter(ILs, factors[idx, ii], color=colors[ii], marker=markers[jj])
            else:
                ax.plot(ILs, factors[idx, ii], color=colors[ii])
                ax.scatter(ILs, factors[idx, ii], color=colors[ii], marker=markers[jj])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.add_artist(ax.legend(handles=legend_shape, loc=2, borderpad=0.3, labelspacing=0.2, handlelength=0.2, handletextpad=0.5, markerscale=0.7, fontsize=8))

    ax.set_xlabel('Ligand Concentration (nM)')
    ax.set_ylabel('Component')
    ax.set_xscale('log')
    ax.set_title('Ligands')

    # Put a legend to the right of the current axis
    ax.legend(loc=3, bbox_to_anchor=(1, 0.5), handletextpad=0.5, handlelength=0.5, framealpha=0.5, markerscale=0.7, fontsize=8)
