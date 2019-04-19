"""
This creates Figure S5. CP decomposition of measured pSTAT data.
"""
import string
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup, plot_cells, set_bounds, import_pstat
from .figure3 import plot_R2X
from ..Tensor_analysis import perform_decomposition


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 7), (2, 2))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])  # Add subplot labels

    ckineConc, _, IL2_data, IL15_data = import_pstat()
    ckineConc = np.round(np.flip(ckineConc).astype(np.double), 5)
    IL2 = np.flip(IL2_data, axis=(0, 1))  # Makes them in both chronological order and ascending stimulation concentration
    IL15 = np.flip(IL15_data, axis=(0, 1))  # Makes them in both chronological order and ascending stimulation concentration
    concat = np.concatenate((IL2, IL15), axis=1)  # Prepare for tensor reshaping
    measured_tensor = np.reshape(concat, (11, 4, 24))
    cell_names = ['Mem Th', 'Naive Th', 'T helper', 'Mem Treg', 'Naive Treg', 'Tregs', 'Mem CD8+', 'Naive CD8+', 'CD3+CD8+', 'NKT', 'NK']

    factors_activity = []
    for jj in range(measured_tensor.shape[2] - 1):
        factors = perform_decomposition(measured_tensor, jj + 1)
        factors_activity.append(factors)
    plot_R2X(ax[0], measured_tensor, factors_activity, n_comps=5)

    n_comps = 2
    factors_activ = factors_activity[n_comps - 1]  # First dimension is cells. Second is time. Third is ligand.
    plot_timepoints(ax[1], factors_activ[1])  # Time is the second dimension in this case because reshaping only correctly did 11*4*24

    plot_cells(ax[2], factors_activ[0], 1, 2, cell_names, ax_pos=1)

    plot_ligands(ax[3], factors_activ[2], 1, 2, ckineConc)
    f.tight_layout()

    return f


def plot_timepoints(ax, factors):
    """Function to put all timepoint curves in one figure."""
    ts = np.array([0.5, 1., 2., 4.]) * 60.

    colors = ['b', 'k', 'r', 'y', 'm', 'g']
    for ii in range(factors.shape[1]):
        ax.plot(ts, factors[:, ii], c=colors[ii], label='Component ' + str(ii + 1))
        ax.scatter(ts[-1], factors[-1, ii], s=12, color='k')

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Component')
    ax.set_title('Time')
    ax.legend()


def plot_ligands(ax, factors, component_x, component_y, IL_treat):
    "This function is to plot the ligand combination dimension of the values tensor."
    markers = ['^', '*']
    n_ligands = len(IL_treat)
    cmap = sns.color_palette("hls", n_ligands)

    legend_shape = [Line2D([0], [0], color='k', marker=markers[0], label='IL-2', linestyle=''),
                    Line2D([0], [0], color='k', label='IL-15', marker=markers[1], linestyle='')]  # only have IL2 and IL15 in the measured pSTAT data

    for ii in range(int(factors.shape[0] / n_ligands)):
        idx = range(ii * n_ligands, (ii + 1) * n_ligands)
        if ii == 0:
            legend = "full"
        else:
            legend = False
        sns.scatterplot(x=factors[idx, component_x - 1], y=factors[idx, component_y - 1], marker=markers[ii], hue=IL_treat, ax=ax, palette=cmap, s=100, legend=legend)
        h, _ = ax.get_legend_handles_labels()
        legend1 = ax.legend(handles=h, loc=2)
        ax.add_artist(legend1)
        legend2 = ax.legend(handles=legend_shape, loc=3)
        ax.add_artist(legend2)

    ax.set_title('Ligands')
    set_bounds(ax, component_x)
