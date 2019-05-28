"""
This creates Figure 3.
"""
import string
import tensorly as tl
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_timepoints, plot_R2X, set_bounds
from ..imports import import_Rexpr
from ..tensor import perform_decomposition
from ..make_tensor import make_tensor, n_lig

cell_dim = 1  # For this figure, the cell dimension is along the second [python index 1].
values, _, mat, _, _ = make_tensor(mut=True)
values = tl.tensor(values)

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 2, 4
    ax, f = getSetup((7.5, 5), (x, y))
    # Blank out for the cartoon
    ax[4].axis('off')
    ax[5].axis('off')

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
    plot_cells(ax[6], factors_activ[1], 2, 3, cell_names, ax_pos=6)

    plot_ligands(ax[3], factors_activ[2], 1, 2, ax_pos=3, n_ligands=n_ligands, mesh=mat, fig=f)
    plot_ligands(ax[7], factors_activ[2], 2, 3, ax_pos=7, n_ligands=n_ligands, mesh=mat, fig=f)

    f.tight_layout()

    return f

def plot_ligands(ax, factors, component_x, component_y, ax_pos, n_ligands, mesh, fig):
    "This function is to plot the ligand combination dimension of the values tensor."
    markers = ['^', '*', '.']
    legend_shape = [Line2D([0], [0], color='k', marker=markers[0], label='IL-2', linestyle=''),
                    Line2D([0], [0], color='k', label='IL-2Ra mut', marker=markers[1], linestyle=''),
                    Line2D([0], [0], color='k', label='IL-2Rb mut', marker=markers[2], linestyle='')]
    hu = np.around(np.sum(mesh[range(int(mesh.shape[0]/n_ligands)), :], axis=1).astype(float), decimals=7)
    norm = LogNorm(vmin=hu.min(), vmax=hu.max())
    cmap = sns.dark_palette("#2eccc0", n_colors=len(hu), reverse=True, as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for ii in range(n_ligands):
        idx = range(ii * int(mesh.shape[0] / n_ligands), (ii + 1) * int(mesh.shape[0] / n_ligands))
        sns.scatterplot(x=factors[idx, component_x-1], y=factors[idx, component_y-1], hue=hu, marker=markers[ii], ax=ax, palette=cmap, s=100, legend=False, hue_norm=LogNorm())

        if ii == 0 and ax_pos == 3:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            a = fig.colorbar(sm, cax=cax)
            a.set_label('Concentration (nM)')
            ax.add_artist(ax.legend(handles=legend_shape, loc=3, borderpad=0.4, labelspacing=0.2, handlelength=0.2, handletextpad=0.5, markerscale=0.7, fontsize=8))

    ax.set_title('Ligands')
    set_bounds(ax, component_x)
