"""
This creates Figure 4.
"""
import string
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, import_samples_2_15, import_samples_4_7, load_cells, plot_conf_int
from ..plot_model_prediction import pstat

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4), mults=[0, 2], multz={0: 2, 2: 2}, empts=[7])

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    data, cell_names = load_cells()
    unkVec_2_15, scales_2_15 = import_samples_2_15()
    unkVec_4_7, scales_4_7 = import_samples_4_7()
    relativeGC(ax[0], unkVec_2_15, unkVec_4_7)
    all_cells(ax[1], data, cell_names, unkVec_2_15[:, 0], scales_2_15[0])
    IL2_receptor_activity(ax[2:5], unkVec_2_15, scales_2_15)

    f.tight_layout(w_pad=0.1, h_pad=1.0)

    return f


def relativeGC(ax, unkVec2, unkVec4):
    """ This function compares the relative complex affinities for GC. The rates included in this violing plot will be k4rev, k10rev, k17rev, k22rev, k27rev, and k33rev. We're currently ignoring k31rev (IL9) and k35rev (IL21) since we don't fit to any of its data. """

    # assign values from unkVec
    k4rev, k5rev, k16rev, k17rev, k22rev, k27rev, k33rev = unkVec2[7, :], unkVec2[8, :], unkVec2[9, :], unkVec2[10, :], unkVec2[11, :], unkVec4[13, :], unkVec4[15, :]

    # back-out k10 with ratio
    k10rev = 12.0 * k5rev / 1.5 # doi:10.1016/j.jmb.2004.04.038

    # add each rate duo as separate column in dataframe
    df = pd.DataFrame({'2·2Rα': k4rev, '2·2Rβ': k5rev, '2·2Rα·2Rβ': k10rev, '15·15Rα': k16rev, '15·2Rβ': k17rev, '15·15Rα·2Rβ': k22rev, '7·7Rα': k27rev, '4·4Rα': k33rev})

    col_list = ["violet", "violet", "violet", "goldenrod", "goldenrod", "goldenrod", "blue", "lightblue"]
    col_list_palette = sns.xkcd_palette(col_list)
    cmap = sns.set_palette(col_list_palette)

    a = sns.violinplot(data=np.log10(df), ax=ax, linewidth=0, bw=10, cmap=cmap, scale='width')
    a.set_xticklabels(a.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.075))
    a.set(title=r"Relative $\gamma_{c}$ affinity", ylabel=r"$\mathrm{log_{10}(\frac{1}{nM * min})}$")

def cell_act(unkVec, cytokC, scale):
    """ Cytokine activity for all IL2 doses for single cell line. """
    pstat5 = pstat()
    K = unkVec.shape[0]
    act = np.zeros((K, cytokC.shape[0]))
    for x, conc in enumerate(cytokC):
            act[:, x] = pstat5.parallelCalc(unkVec.T, 0, conc)

    # normalize to scaling constant and maximal activity for each row
    for num in range(act.shape[0]):
        act[num] = act[num] / (act[num] + scale[num])
        act[num] = act[num] / np.max(act[num])

    return act

def all_cells(ax, cell_data, cell_names, unkVec, scale):
    """ Loops through all cell types and calculates activities. """
    cell_data = cell_data.values    # convert to numpy array
    PTS = 60    # number of cytokine concentrations that are used
    cytokC = np.logspace(-5, 0, PTS)
    numCells = cell_data.shape[1] - 1   # first column is receptor names

    colors = cm.rainbow(np.linspace(0, 1, numCells))

    newVec = np.tile(unkVec, (numCells, 1)) # copy unkVec numCells times to create a 2D array
    newVec[:, 22:30] = cell_data[:, 1:].T # place cell data into newVec
    scaleVec = np.repeat(scale, numCells) # create an array of scale to match size of newVec

    act = cell_act(newVec, cytokC, scaleVec) # run simulations

    # plot results
    for ii in range(act.shape[0]):
        ax.plot(np.log10(cytokC), act[ii], label=cell_names[ii], c=colors[ii])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.2))
    ax.set(title="Cell Response to IL-2", ylabel="pSTAT5 (% of max)", xlabel=r'IL-2 concentration (log$_{10}$[nM])')

def IL2_receptor_activity(ax, unkVec, scales):
    """ Shows how IL2-pSTAT dose response curves change with receptor expression rates. """
    PTS = 30 # number of cytokine concentrations
    split = 50 # number of rows used from unkVec
    cytokC = np.logspace(-3.3, 2.7, PTS)
    factors = np.array([0.01, 0.1, 1, 10, 100]) # factors that we multiply the receptor expression rates by
    y_max = 100.

    # create separate plot for each receptor
    for r in range(0,3):
        newVec = np.tile(unkVec[:, 0:split], (1, len(factors))) # copy the first 50 rows of unkVec 5 times (corresponds with factors)
        newScales = np.squeeze(np.tile(scales[0:split], (len(factors), 1))) # copy the first 50 rows of scales 5 times

        # multiply receptor expression rate for each section of newVec
        newVec[22+r, 0:split] *= factors[0]
        newVec[22+r, split:(2*split)] *= factors[1]
        newVec[22+r, (2*split):(3*split)] *= factors[2]
        newVec[22+r, (3*split):(4*split)] *= factors[3]
        newVec[22+r, (4*split):(5*split)] *= factors[4]

        # calculate activities in parallel
        output = cell_act(newVec.T, cytokC, newScales).T * y_max

        plot_conf_int(ax[r], np.log10(cytokC), output[:, 0:split], "royalblue", "0.01x")
        plot_conf_int(ax[r], np.log10(cytokC), output[:, split:(2*split)], "navy", "0.1x")
        plot_conf_int(ax[r], np.log10(cytokC), output[:, (2*split):(3*split)], "darkviolet", "1x")
        plot_conf_int(ax[r], np.log10(cytokC), output[:, (3*split):(4*split)], "deeppink", "10x")
        plot_conf_int(ax[r], np.log10(cytokC), output[:, (4*split):(5*split)], "red", "100x")
        ax[r].set(xlabel=r'IL-2 concentration (log$_{10}$[nM])', ylabel="Total pSTAT")

    ax[0].set_title("IL-2Rα")
    ax[1].set_title("IL-2Rβ")
    ax[2].set_title(r'$\gamma_{c}$')
    ax[2].legend(loc='upper left', bbox_to_anchor=(1.05, 0.75))
