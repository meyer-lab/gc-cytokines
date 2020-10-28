"""
This creates Figure S5. Full panel of measured vs simulated for IL-2 and IL-15.
"""
import numpy as np
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, plot_conf_int, plot_scaled_pstat, calc_dose_response
from ..imports import import_Rexpr, import_samples_2_15, import_pstat


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 5))

    subplotLabel(ax)

    plot_exp_v_pred(ax)
    for jj, axes in enumerate(ax):
        if jj < 10:
            axes.set_xlabel(r"[IL-2] Concentration (nM)", fontsize=6)
            axes.xaxis.set_tick_params(labelsize=7)
        else:
            axes.set_xlabel(r"[IL-15] Concentration (nM)", fontsize=6)
            axes.xaxis.set_tick_params(labelsize=7)

    return f


def plot_exp_v_pred(ax, cell_subset=None):
    """ Perform main routine for plotting all the cell predictions vs. experimental values.
    The default argument of cell_subset is an empty list which ends up plotting all 10 cell types;
    if one wishes to one plot a subset of cells they must be noted in list format. """
    _, receptor_data, _ = import_Rexpr()
    unkVec_2_15, scales = import_samples_2_15(N=100)  # use all rates
    ckineConc, cell_names_pstat, IL2_data, IL15_data, _ = import_pstat()

    # Scale all the data down so we don't have a bunch of zeros on our axes
    IL2_data = IL2_data / 1000.0
    IL15_data = IL15_data / 1000.0

    if cell_subset is None:
        cell_subset = []

    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    axis = 0
    shift = 10 if cell_subset == [] else len(cell_subset)  # there are 10 cells if no subset is given

    IL2_activity, IL15_activity, _ = calc_dose_response(cell_names_pstat, unkVec_2_15, scales, receptor_data, tps, ckineConc, IL2_data, IL15_data)

    for m, name in enumerate(cell_names_pstat):
        if name in cell_subset or cell_subset == []:  # if a subset is provided only plot the listed names
            if axis == 0:
                plot_dose_response(ax[axis], ax[axis + shift], IL2_activity[m, :, :, :], IL15_activity[m, :, :, :], name, tps, ckineConc, legend=True)
            else:
                plot_dose_response(ax[axis], ax[axis + shift], IL2_activity[m, :, :, :], IL15_activity[m, :, :, :], name, tps, ckineConc)
            plot_scaled_pstat(ax[axis], ckineConc.astype(np.float), IL2_data[(m * 4): ((m + 1) * 4)])
            plot_scaled_pstat(ax[axis + shift], ckineConc.astype(np.float), IL15_data[(m * 4): ((m + 1) * 4)])
            axis = axis + 1


def plot_dose_response(ax2, ax15, IL2_activity, IL15_activity, cell_type, tps, cytokC, legend=False):
    """ Plots both IL2 and IL15 activity in different plots where each plot has multiple timepoints and cytokine concentrations. """
    colors = cm.rainbow(np.linspace(0, 1, tps.size))

    # plot the values with each time as a separate color
    for tt in range(tps.size):
        if legend:
            plot_conf_int(ax2, cytokC.astype(np.float), IL2_activity[:, :, tt], colors[tt], (tps[tt] / 60.0).astype(str))
            ax2.legend(title="time (hours)", prop={"size": 6}, title_fontsize=6, loc="upper left")
        else:
            plot_conf_int(ax2, cytokC.astype(np.float), IL2_activity[:, :, tt], colors[tt])

        plot_conf_int(ax15, cytokC.astype(np.float), IL15_activity[:, :, tt], colors[tt])

    # plots for input cell type
    ax2.set(ylabel="pSTAT5", title=cell_type)
    ax2.set_xlabel(r"[IL-2] Concentration (nM)", fontsize=8)
    ax15.set(ylabel="pSTAT5", title=cell_type)
    ax15.set_xlabel(r"[IL-15] Concentration (nM)", fontsize=8)
    ax2.set_xscale("log")
    ax15.set_xscale("log")
    ax2.set_xlim(10e-5, 10e1)
    ax15.set_xlim(10e-5, 10e1)
    ax2.set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
    ax15.set_xticks(np.array([10e-5, 10e-3, 10e-1, 10e1]))
