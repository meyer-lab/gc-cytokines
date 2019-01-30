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
from ..model import getTotalActiveSpecies, runCkineU

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4), mults=[0, 2], multz={0: 2, 2: 2}, empts=[7])

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    data, cell_names = load_cells()
    unkVec_2_15 = import_samples_2_15()
    unkVec_4_7, scales = import_samples_4_7()
    relativeGC(ax[0], unkVec_2_15, unkVec_4_7)
    all_cells(ax[1], data, cell_names, unkVec_2_15[:, 0])
    IL2_receptor_activity(ax[2:5], unkVec_2_15)

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

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.2))
    ax.set(title="Cell Response to IL-2", ylabel="pSTAT5 (% of max)", xlabel=r'IL-2 concentration (log$_{10}$[nM])')

def IL2_receptor_activity(ax, unkVec):
    """ Shows how IL2-pSTAT dose response curves change with receptor expression rates. """
    PTS = 30
    cytokC = np.logspace(-3.3, 2.7, PTS)
    y_max = 100.
    activity = np.zeros((PTS, 50, 5, 3))
    factors = np.array([0.01, 0.1, 1, 10, 100]) # factors that we multiply the receptor expression rates by
    for r in range(0,3):
        for n in range(factors.size):
            unkVec2 = unkVec.copy()
            unkVec2[22+r] *= factors[n]  # multiply receptor expression rate by factor
            for ii in range(0,50):
                output = rec_act_calc(unkVec2[:, ii], cytokC) * y_max
                activity[:, ii, n, r] = output[0:PTS]

        plot_conf_int(ax[r], np.log10(cytokC), activity[:,:,0,r], "royalblue", "0.01x")
        plot_conf_int(ax[r], np.log10(cytokC), activity[:,:,1,r], "navy", "0.1x")
        plot_conf_int(ax[r], np.log10(cytokC), activity[:,:,2,r], "darkviolet", "1x")
        plot_conf_int(ax[r], np.log10(cytokC), activity[:,:,3,r], "deeppink", "10x")
        plot_conf_int(ax[r], np.log10(cytokC), activity[:,:,4,r], "red", "100x")
        ax[r].set(xlabel=r'IL-2 concentration (log$_{10}$[nM])', ylabel="Total pSTAT")

    ax[0].set_title("IL-2Rα")
    ax[1].set_title("IL-2Rβ")
    ax[2].set_title(r'$\gamma_{c}$')
    ax[2].legend(loc='upper left', bbox_to_anchor=(1.05, 0.75))

def rec_act_singleCalc(unkVec, cytokine, conc):
    """ Calculates pSTAT activity over time for one condition. """
    unkVec = unkVec.copy()
    unkVec[cytokine] = conc
    ts = np.array([500.])
    returnn, retVal = runCkineU(ts, unkVec)

    assert retVal >= 0
    activity = getTotalActiveSpecies().astype(np.float64)
    return np.dot(returnn, activity)

def rec_act_calc(unkVec, cytokC):
    ''' Finds pSTAT activity for all cytokine concentrations given. '''
    actVec = np.fromiter((rec_act_singleCalc(unkVec, 0, x) for x in cytokC), np.float64)

    return actVec
