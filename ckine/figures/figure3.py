"""
This creates Figure 3.
"""
import logging
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
import matplotlib.cm as cm
import string
import tensorly as tl
import seaborn as sns
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints, plot_R2X, set_bounds
from ..imports import import_Rexpr
from ..tensor import perform_decomposition
from ..make_tensor import make_tensor, n_lig

cell_dim = 1  # For this figure, the cell dimension is along the second [python index 1].
values, _, mat, _, _ = make_tensor()
values = tl.tensor(values)
logging.info("Done constructing tensor.")


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 7), (3, 4), multz={0: 1, 4: 2})

    logging.info("Starting decomposition.")
    data, numpy_data, cell_names = import_Rexpr()
    factors_activity = []
    for jj in range(len(mat) - 1):
        factors = perform_decomposition(values, jj + 1, cell_dim)
        factors_activity.append(factors)
    logging.info("Decomposition finished.")

    n_comps = 3
    factors_activ = factors_activity[n_comps - 1]

    # Start plotting
    PCA_receptor(ax[1:3], cell_names, numpy_data)
    catplot_receptors(ax[0], data)

    # Blank out for the cartoon
    ax[3].axis('off')

    plot_R2X(ax[4], values, factors_activity, n_comps=5, cells_dim=cell_dim)

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])  # Add subplot labels

    plot_timepoints(ax[5], tl.to_numpy(factors_activ[0]))  # Change final input value depending on need

    plot_cells(ax[6], tl.to_numpy(factors_activ[1]), 1, 2, cell_names, ax_pos=5)
    plot_cells(ax[7], tl.to_numpy(factors_activ[1]), 2, 3, cell_names, ax_pos=6)

    plot_ligands(ax[8], tl.to_numpy(factors_activ[2]), n_ligands=4, fig=3, mesh=mat)

    return f


def catplot_receptors(ax, data):
    """Plot Bar graph for Receptor Expression Data. """
    sns.catplot(x="Cell Type", y="Count", hue="Receptor", data=data, ci=68, ax=ax)
    ax.set_ylabel("Surface Receptor [# / cell]")
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=25, rotation_mode="anchor", ha="right",
                       position=(0, 0.02), fontsize=7.5)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)


def PCA_receptor(ax, cell_names, data):
    """Plot PCA scores and loadings for Receptor Expression Data. """
    pca = PCA(n_components=2)
    data = stats.zscore(data.astype(float), axis=1)
    scores = pca.fit(data).transform(data)  # 11 cells by n_comp
    loadings = pca.components_  # n_comp by 7 receptors
    expVar = pca.explained_variance_ratio_

    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '4', 'H', 'P', '*', 'D', 's', 'X']
    markersReceptors = ['^', '4', 'P', '*', 'D']
    labelReceptors = ['IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra']  # 'IL9R', 'IL4Ra', 'IL21Ra']

    for ii in range(scores.shape[0]):
        ax[0].scatter(scores[ii, 0], scores[ii, 1], c=[colors[ii]], marker=markersCells[ii], label=cell_names[ii])

    for jj in range(loadings.shape[1]):
        ax[1].scatter(loadings[0, jj], loadings[1, jj], marker=markersReceptors[jj], label=labelReceptors[jj])

    x_max1 = np.max(np.absolute(np.asarray(ax[0].get_xlim()))) * 1.1
    y_max1 = np.max(np.absolute(np.asarray(ax[0].get_ylim()))) * 1.1

    x_max2 = np.max(np.absolute(np.asarray(ax[1].get_xlim()))) * 1.1
    y_max2 = np.max(np.absolute(np.asarray(ax[1].get_ylim()))) * 1.1

    ax[0].set_xlim(-x_max1, x_max1)
    ax[0].set_ylim(-y_max1, y_max1)
    ax[0].set_xlabel('PC1 (' + str(round(expVar[0] * 100, 2)) + '%)')
    ax[0].set_ylabel('PC2 (' + str(round(expVar[1] * 100, 2)) + '%)')
    ax[0].set_title('Scores')
    ax[0].legend(loc='upper left', borderpad=0.35, labelspacing=0.1, handlelength=0.2, handletextpad=0.5, markerscale=0.65, fontsize=8, fancybox=True, framealpha=0.5)
    set_bounds(ax[0], 1)

    ax[1].set_xlim(-x_max2, x_max2)
    ax[1].set_ylim(-y_max2, y_max2)
    ax[1].set_xlabel('PC1 (' + str(round(expVar[0] * 100, 2)) + '%)')
    ax[1].set_ylabel('PC2 (' + str(round(expVar[1] * 100, 2)) + '%)')
    ax[1].set_title('Loadings')
    ax[1].legend()
    set_bounds(ax[1], 1)
