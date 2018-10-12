"""
This creates Figure 4.
"""
import string
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, import_samples_2_15, import_samples_4_7, load_cells
from ..plot_model_prediction import pstat

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4), mults=[0, 2], multz={0: 2, 2: 2})

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    f.tight_layout()

    data, cell_names = load_cells()
    unkVec_2_15 = import_samples_2_15()
    unkVec_4_7, scales = import_samples_4_7()
    all_cells(ax[1], data, cell_names, unkVec_2_15[:, 0])
    relativeGC(ax[0], unkVec_2_15, unkVec_4_7)


    return f


def relativeGC(ax, unkVec2, unkVec4):
    """ This function compares the relative complex affinities for GC. The rates included in this violing plot will be k4rev, k10rev, k17rev, k22rev, k27rev, and k33rev. We're currently ignoring k31rev (IL9) and k35rev (IL21) since we don't fit to any of its data. """

    # assign values from unkVec
    k4rev, k5rev, k16rev, k17rev, k22rev, k27rev, k33rev = unkVec2[7, :], unkVec2[8, :], unkVec2[9, :], unkVec2[10, :], unkVec2[11, :], unkVec4[13, :], unkVec4[15, :]

    # back-out k10 with ratio
    k10rev = 12.0 * k5rev / 1.5 # doi:10.1016/j.jmb.2004.04.038

    # add each rate duo as separate column in dataframe
    df = pd.DataFrame({'2-2Ra': k4rev, '2-2Rb': k5rev, '2-2Ra-2Rb': k10rev, '15-15Ra': k16rev, '15-2Rb': k17rev, '15-15Ra-2Rb': k22rev, '7-7Ra': k27rev, '4-4Ra': k33rev})

    col_list = ["violet", "violet", "violet", "goldenrod", "goldenrod", "goldenrod", "blue", "lightblue"]
    col_list_palette = sns.xkcd_palette(col_list)
    cmap = sns.set_palette(col_list_palette)

    a = sns.violinplot(data=np.log10(df), ax=ax, linewidth=0, bw=10, cmap=cmap, scale='width')
    a.set_xticklabels(a.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.075))
    a.set(title="Relative gc affinity", ylabel="log10 of 1/nM/min")

def single_cell_act(unkVec, cytokC):
    """ Cytokine activity for all IL2 doses for single cell line. """
    pstat5 = pstat()
    act = np.zeros((cytokC.shape[0]))
    act = np.fromiter((pstat5.singleCalc(unkVec, 0, x) for x in cytokC), np.float64)
    return act / np.max(act)    # normalize to maximal activity


def all_cells(ax, cell_data, cell_names, unkVec):
    """ Loops through all cell types and calculates activities. """
    cell_data = cell_data.values    # convert to numpy array
    PTS = 60    # number of cytokine concentrations that are used
    cytokC = np.logspace(-5, 0, PTS)
    numCells = cell_data.shape[1] - 1   # first column is receptor names

    colors = cm.rainbow(np.linspace(0, 1, numCells))

    for ii in range(0, numCells):       # for all cell types
        unkVec[22:30] = cell_data[:, ii+1]  # place cell data into unkVec
        act = single_cell_act(unkVec, cytokC)
        ax.plot(np.log10(cytokC), act, label=cell_names[ii], c=colors[ii])
        # if (act[0] > 0.1):
        #   print(cell_names[ii]) # tells us that proB_FrBC_BM and T_DP_Th cells respond at the lowest IL2 conc.

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.2))
    ax.set(title="Cell Response to IL-2", ylabel="Relative pSTAT5 activity (% x 1)", xlabel="log10 IL-2 conc. (nM)")