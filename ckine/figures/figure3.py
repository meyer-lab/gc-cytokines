"""
This creates Figure 3.
"""
import string
import logging
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.cm as cm
import tensorly as tl
import seaborn as sns
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints, plot_R2X, set_bounds
from ..imports import import_Rexpr
from ..tensor import perform_decomposition, z_score_values
from ..make_tensor import make_tensor, tensor_time

cell_dim = 1  # For this figure, the cell dimension is along the second [python index 1].
values, _, mat, _, _ = make_tensor()
values[:, :, 36:48] /= 7.0 # IL-7 just seems to have larger values across the board, so shrink a bit
values = z_score_values(tl.tensor(values), cell_dim)
logging.info("Done constructing tensor.")

logging.info("Starting decomposition.")
data, numpy_data, cell_names = import_Rexpr()
factors_activity = []
for jj in range(4):
    factors = perform_decomposition(values, jj + 1)
    factors_activity.append(factors)
logging.info("Decomposition finished.")
n_pred_comps = 3 #Placed here to be also used by Figure 5

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 6), (3, 4), multz={0: 1, 4: 1})

    factors_activ = factors_activity[n_pred_comps - 1]

    # Start plotting
    PCA_receptor(ax[1:3], cell_names, numpy_data)
    catplot_receptors(ax[0], data)

    # Blank out for the cartoon
    ax[3].axis('off')
    ax[4].axis('off')

    plot_R2X(ax[5], values, factors_activity)

    # Add subplot labels
    axLabel = ax.copy()
    del axLabel[4]
    for ii, item in enumerate(axLabel):
        subplotLabel(item, string.ascii_uppercase[ii], hstretch=(1.5 if ii == 3 else 1))  # Add subplot labels

    plot_timepoints(ax[6], tensor_time, tl.to_numpy(factors_activ[0]))

    plot_cells(ax[7], tl.to_numpy(factors_activ[1]), 1, 2, cell_names)
    plot_cells(ax[8], tl.to_numpy(factors_activ[1]), 2, 3, cell_names)

    legend = ax[7].get_legend()
    labels = (x.get_text() for x in legend.get_texts())
    ax[4].legend(legend.legendHandles, labels, loc='lower right')

    ax[7].get_legend().remove()
    ax[8].get_legend().remove()

    plot_ligands(ax[9], tl.to_numpy(factors_activ[2]), ligand_names=['IL-2', 'mut IL-2', 'IL-15', 'IL-7'])

    return f

def catplot_receptors(ax, datas):
    """Plot Bar graph for Receptor Expression Data. """
    sns.catplot(x="Cell Type", y="Count", hue="Receptor", data=datas, ci=68, ax=ax, kind="point", join=False, scale=0.5, dodge=True, estimator=sp.stats.gmean)
    ax.set_ylabel("Surface Receptor [# / cell]")
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=25, rotation_mode="anchor", ha="right",
                       position=(0, 0.02), fontsize=7.5)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)


def PCA_receptor(ax, cell_name, datas):
    """Plot PCA scores and loadings for Receptor Expression Data. """
    pca = PCA(n_components=2)
    recep_data = stats.zscore(datas[:, [0, 1, 2, 4]], axis=0)
    scores = pca.fit_transform(recep_data)  # 10 cells by n_comp
    loadings = pca.components_  # n_comp by 7 receptors
    expVar = pca.explained_variance_ratio_

    colors = cm.rainbow(np.linspace(0, 1, len(cell_name)))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '4', 'H', 'P', '*', 'D', 's', 'X']

    for ii in range(scores.shape[0]):
        ax[0].scatter(scores[ii, 0], scores[ii, 1], c=[colors[ii]], marker=markersCells[ii], label=cell_name[ii])

    for kk in range(loadings.shape[1]):
        if kk == 3:
            ax[1].scatter(0, 0, s=16, label='IL-15Ra')
        ax[1].scatter(loadings[0, kk], loadings[1, kk], s=16)

    ax[0].set_title('Scores')
    set_bounds(ax[0])
    ax[0].set_xlabel('PC1 (' + str(round(expVar[0] * 100)) + '%)')
    ax[0].set_ylabel('PC2 (' + str(round(expVar[1] * 100)) + '%)')

    ax[1].set_title('Loadings')
    set_bounds(ax[1])
    ax[1].set_xlabel('PC1 (' + str(round(expVar[0] * 100)) + '%)')
    ax[1].set_ylabel('PC2 (' + str(round(expVar[1] * 100)) + '%)')
