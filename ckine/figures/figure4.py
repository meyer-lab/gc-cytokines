"""
This creates Figure 4.
"""
import string
import numpy as np
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, import_samples_2_15, plot_conf_int, import_Rexpr
from ..plot_model_prediction import pstat
from ..model import runCkineUP, getTotalActiveSpecies, receptor_expression


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 4), empts=[13, 14, 15])

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    data_Visterra, cell_names_Visterra = import_Rexpr()
    unkVec_2_15, _ = import_samples_2_15(N=100)

    # IL2_receptor_activity(ax[2:5], unkVec_2_15, scales_2_15)
    for i in range(data_Visterra.shape[0]):
        if i == (data_Visterra.shape[0] - 1):  # only plot the legend for the last entry
            IL2_dose_response(ax[i], unkVec_2_15, cell_names_Visterra[i], data_Visterra[i], legend=True)
        else:
            IL2_dose_response(ax[i], unkVec_2_15, cell_names_Visterra[i], data_Visterra[i])

    f.tight_layout(w_pad=0.1, h_pad=1.0)

    return f


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


def IL2_receptor_activity(ax, unkVec, scales):
    """ Shows how IL2-pSTAT dose response curves change with receptor expression rates. """
    PTS = 30  # number of cytokine concentrations
    split = 50  # number of rows used from unkVec
    cytokC = np.logspace(-3.3, 2.7, PTS)
    factors = np.array([0.01, 0.1, 1, 10, 100])  # factors that we multiply the receptor expression rates by
    y_max = 100.

    # create separate plot for each receptor
    for r in range(0, 3):
        newVec = np.tile(unkVec[:, 0:split], (1, len(factors)))  # copy the first 50 rows of unkVec 5 times (corresponds with factors)
        newScales = np.squeeze(np.tile(scales[0:split], (len(factors), 1)))  # copy the first 50 rows of scales 5 times

        # multiply receptor expression rate for each section of newVec
        newVec[22 + r, 0:split] *= factors[0]
        newVec[22 + r, split:(2 * split)] *= factors[1]
        newVec[22 + r, (2 * split):(3 * split)] *= factors[2]
        newVec[22 + r, (3 * split):(4 * split)] *= factors[3]
        newVec[22 + r, (4 * split):(5 * split)] *= factors[4]

        # calculate activities in parallel
        output = cell_act(newVec.T, cytokC, newScales).T * y_max

        plot_conf_int(ax[r], np.log10(cytokC), output[:, 0:split], "royalblue", "0.01x")
        plot_conf_int(ax[r], np.log10(cytokC), output[:, split:(2 * split)], "navy", "0.1x")
        plot_conf_int(ax[r], np.log10(cytokC), output[:, (2 * split):(3 * split)], "darkviolet", "1x")
        plot_conf_int(ax[r], np.log10(cytokC), output[:, (3 * split):(4 * split)], "deeppink", "10x")
        plot_conf_int(ax[r], np.log10(cytokC), output[:, (4 * split):(5 * split)], "red", "100x")
        ax[r].set(xlabel=r'IL-2 concentration (log$_{10}$[nM])', ylabel="Total pSTAT")

    ax[0].set_title("IL-2Rα")
    ax[1].set_title("IL-2Rβ")
    ax[2].set_title(r'$\gamma_{c}$')
    ax[2].legend(loc='upper left', bbox_to_anchor=(1.05, 0.75))


def IL2_dose_response(ax, unkVec, cell_type, cell_data, legend=False):
    """ Shows activity for a given cell type at various IL2 concentrations """
    tps = np.array([15., 30., 60., 240.])
    PTS = 6  # number of cytokine concentrations
    cytokC = np.logspace(-4.0, 2.0, PTS)  # vary cytokine concentration from 1 pm to 100 nm
    colors = cm.rainbow(np.linspace(0, 1, tps.size))

    rxntfr = unkVec.T.copy()
    split = rxntfr.shape[0]  # number of parameter sets used (& thus the number of yOut replicates)
    total_activity = np.zeros((PTS, split, tps.size))

    # loop for each IL2 concentration
    for i in range(PTS):
        for ii in range(rxntfr.shape[0]):
            rxntfr[ii, 0] = cytokC[i]
            # updates rxntfr for receptor expression for IL2Ra, IL2Rb, gc
            rxntfr[ii, 22] = receptor_expression(cell_data[0], rxntfr[ii, 17], rxntfr[ii, 20], rxntfr[ii, 19], rxntfr[ii, 21])
            rxntfr[ii, 23] = receptor_expression(cell_data[1], rxntfr[ii, 17], rxntfr[ii, 20], rxntfr[ii, 19], rxntfr[ii, 21])
            rxntfr[ii, 24] = receptor_expression(cell_data[2], rxntfr[ii, 17], rxntfr[ii, 20], rxntfr[ii, 19], rxntfr[ii, 21])
        yOut, retVal = runCkineUP(tps, rxntfr)
        assert retVal >= 0  # make sure solver is working
        activity = np.dot(yOut, getTotalActiveSpecies().astype(np.float))
        for j in range(split):
            total_activity[i, j, :] = activity[(4 * j):((j + 1) * 4)]  # save the activity from this concentration for all 4 tps

    # plot the values with each time as a separate color
    for tt in range(tps.size):
        plot_conf_int(ax, np.log10(cytokC), total_activity[:, :, tt], colors[tt], (tps[tt]).astype(str))

    # plots for input cell type
    ax.set(xlabel=r'IL-2 concentration (log$_{10}$[nM])', ylabel='Activity', title=cell_type)
    if legend is True:
        ax.legend(title='time (min)', loc='center left', borderaxespad=10.)
