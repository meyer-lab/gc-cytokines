"""
This creates Figure S5. Full panel of measured vs simulated for IL-2 and IL-15.
"""
import string
import numpy as np
import matplotlib.cm as cm
from scipy.optimize import least_squares
from .figureCommon import subplotLabel, getSetup, plot_conf_int, plot_scaled_pstat
from ..model import runCkineUP, getTotalActiveSpecies, receptor_expression
from ..imports import import_Rexpr, import_pstat, import_samples_2_15


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 5))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    _, receptor_data, cell_names_receptor = import_Rexpr()
    unkVec_2_15, scales = import_samples_2_15(N=100)  # use all rates
    ckineConc, cell_names_pstat, IL2_data, IL15_data = import_pstat()

    # Scale all the data down so we don't have a bunch of zeros on our axes
    IL2_data = IL2_data / 1000.0
    IL15_data = IL15_data / 1000.0

    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    axis = 0

    for i, _ in enumerate(cell_names_pstat):
        # plot matching experimental and predictive pSTAT data for the same cell type
        assert cell_names_pstat[i] == cell_names_receptor[i]
        IL2_activity, IL15_activity = calc_dose_response(unkVec_2_15, scales, receptor_data[i], tps, ckineConc, IL2_data[(i * 4) : ((i + 1) * 4)], IL15_data[(i * 4) : ((i + 1) * 4)])
        if axis == 9:  # only plot the legend for the last entry
            plot_dose_response(ax[axis], ax[axis + 10], IL2_activity, IL15_activity, cell_names_receptor[i], tps, ckineConc, legend=True)
        else:
            plot_dose_response(ax[axis], ax[axis + 10], IL2_activity, IL15_activity, cell_names_receptor[i], tps, ckineConc)
        plot_scaled_pstat(ax[axis], np.log10(ckineConc.astype(np.float)), IL2_data[(i * 4) : ((i + 1) * 4)])
        plot_scaled_pstat(ax[axis + 10], np.log10(ckineConc.astype(np.float)), IL15_data[(i * 4) : ((i + 1) * 4)])
        axis = axis + 1

    return f


def calc_dose_response(unkVec, scales, cell_data, tps, cytokC, exp_data_2, exp_data_15):
    """ Calculates activity for a given cell type at various cytokine concentrations and timepoints. """
    PTS = cytokC.shape[0]  # number of cytokine concentrations

    rxntfr2 = unkVec.T.copy()
    total_activity2 = np.zeros((PTS, rxntfr2.shape[0], tps.size))
    total_activity15 = total_activity2.copy()

    # updates rxntfr for receptor expression for IL2Ra, IL2Rb, gc
    rxntfr2[:, 22] = receptor_expression(cell_data[0], rxntfr2[:, 17], rxntfr2[:, 20], rxntfr2[:, 19], rxntfr2[:, 21])
    rxntfr2[:, 23] = receptor_expression(cell_data[1], rxntfr2[:, 17], rxntfr2[:, 20], rxntfr2[:, 19], rxntfr2[:, 21])
    rxntfr2[:, 24] = receptor_expression(cell_data[2], rxntfr2[:, 17], rxntfr2[:, 20], rxntfr2[:, 19], rxntfr2[:, 21])
    rxntfr2[:, 25] = 0.0  # We never observed any IL-15Ra

    rxntfr15 = rxntfr2.copy()

    # loop for each IL2 concentration
    for i in range(PTS):
        rxntfr2[:, 0] = rxntfr15[:, 1] = cytokC[i]  # assign concs for each cytokine

        # handle case of IL-2
        yOut = runCkineUP(tps, rxntfr2)
        activity2 = np.dot(yOut, getTotalActiveSpecies().astype(np.float))
        # handle case of IL-15
        yOut = runCkineUP(tps, rxntfr15)
        activity15 = np.dot(yOut, getTotalActiveSpecies().astype(np.float))

        total_activity2[i, :, :] = np.reshape(activity2, (-1, 4))  # save the activity from this concentration for all 4 tps
        total_activity15[i, :, :] = np.reshape(activity15, (-1, 4))  # save the activity from this concentration for all 4 tps

    # scale receptor/cell measurements to pSTAT activity for each sample
    for j in range(len(scales)):
        scale1, scale2 = optimize_scale(total_activity2[:, j, :], total_activity15[:, j, :], exp_data_2, exp_data_15)  # find optimal constants
        total_activity2[:, j, :] = scale2 * total_activity2[:, j, :] / (total_activity2[:, j, :] + scale1)  # adjust activity for this sample
        total_activity15[:, j, :] = scale2 * total_activity15[:, j, :] / (total_activity15[:, j, :] + scale1)  # adjust activity for this sample

    return total_activity2, total_activity15


def plot_dose_response(ax2, ax15, IL2_activity, IL15_activity, cell_type, tps, cytokC, legend=False):
    """ Plots both IL2 and IL15 activity in different plots where each plot has multiple timepoints and cytokine concentrations. """
    colors = cm.rainbow(np.linspace(0, 1, tps.size))

    # plot the values with each time as a separate color
    for tt in range(tps.size):
        plot_conf_int(ax2, np.log10(cytokC.astype(np.float)), IL2_activity[:, :, tt], colors[tt])  # never a legend for IL-2
        if legend:
            plot_conf_int(ax15, np.log10(cytokC.astype(np.float)), IL15_activity[:, :, tt], colors[tt], (tps[tt] / 60.0).astype(str))
            ax15.legend(title="time (hours)")
        else:
            plot_conf_int(ax15, np.log10(cytokC.astype(np.float)), IL15_activity[:, :, tt], colors[tt])

    # plots for input cell type
    ax2.set(xlabel=r"[IL-2] (log$_{10}$[nM])", ylabel="Activity", title=cell_type)
    ax15.set(xlabel=r"[IL-15] (log$_{10}$[nM])", ylabel="Activity", title=cell_type)


def optimize_scale(model_act2, model_act15, exp_act2, exp_act15):
    """ Formulates the optimal scale to minimize the residual between model activity predictions and experimental activity measurments for a given cell type. """
    exp_act2 = exp_act2.T  # transpose to match model_act
    exp_act15 = exp_act15.T

    # scaling factors are sigmoidal and linear, respectively
    guess = np.array([100.0, np.mean(exp_act2 + exp_act15) / np.mean(model_act2 + model_act15)])

    def calc_res(sc):
        """ Calculate the residuals. This is the function we minimize. """
        scaled_act2 = sc[1] * model_act2 / (model_act2 + sc[0])
        scaled_act15 = sc[1] * model_act15 / (model_act15 + sc[0])
        err = np.hstack((exp_act2 - scaled_act2, exp_act15 - scaled_act15))
        return np.reshape(err, (-1,))

    # find result of minimization where both params are >= 0
    res = least_squares(calc_res, guess, bounds=(0.0, np.inf))
    return res.x
