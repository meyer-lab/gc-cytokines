"""
This creates Figure S5. Full panel of measured vs simulated for IL-2 and IL-15.
"""
import string
import numpy as np
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, plot_conf_int, plot_scaled_pstat, calc_dose_response, grouped_scaling
from ..imports import import_Rexpr, import_samples_2_15, import_pstat


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 5))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    plot_exp_v_pred(ax)

    return f


def plot_exp_v_pred(ax, cell_subset=None):
    """ Perform main routine for plotting all the cell predictions vs. experimental values.
    The default argument of cell_subset is an empty list which ends up plotting all 10 cell types;
    if one wishes to one plot a subset of cells they must be noted in list format. """
    _, receptor_data, cell_names_receptor = import_Rexpr()
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

    total_activity2 = np.zeros((len(cell_names_pstat), len(ckineConc), len(scales), len(tps)))
    total_activity15 = total_activity2.copy()
    for i, name in enumerate(cell_names_pstat):
        if name in cell_subset or cell_subset == []:  # if a subset is provided only plot the listed names
            assert cell_names_pstat[i] == cell_names_receptor[i]
            IL2_activity, IL15_activity = calc_dose_response(unkVec_2_15, scales, receptor_data[i], tps, ckineConc, IL2_data[(i * 4): ((i + 1) * 4)], IL15_data[(i * 4): ((i + 1) * 4)])
            total_activity2[i, :, :, :] = IL2_activity
            total_activity15[i, :, :, :] = IL15_activity

    scale = grouped_scaling(scales, cell_names_pstat, IL2_data, IL15_data, total_activity2, total_activity15)

    cell_groups = [['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+', 'Naive CD8+', 'Mem CD8+']]

    for j, cells in enumerate(cell_groups):
        for k, cell in enumerate(cell_names_pstat):
            if cell in cells:
                for l in range(scale.shape[2]):
                    total_activity2[k, :, l, :] = scale[j, 1, l] * total_activity2[k, :, l, :] / (total_activity2[k, :, l, :] + scale[j, 0, l])  # adjust activity for this sample
                    total_activity15[k, :, l, :] = scale[j, 1, l] * total_activity15[k, :, l, :] / (total_activity15[k, :, l, :] + scale[j, 0, l])  # adjust activity for this sample

    for m, name in enumerate(cell_names_pstat):
        if axis == shift - 1:  # only plot the legend for the last entry
            plot_dose_response(ax[axis], ax[axis + shift], total_activity2[m, :, :, :], total_activity15[m, :, :, :], name, tps, ckineConc, legend=True)
        else:
            plot_dose_response(ax[axis], ax[axis + shift], total_activity2[m, :, :, :], total_activity15[m, :, :, :], name, tps, ckineConc)
        plot_scaled_pstat(ax[axis], np.log10(ckineConc.astype(np.float)), IL2_data[(m * 4): ((m + 1) * 4)])
        plot_scaled_pstat(ax[axis + shift], np.log10(ckineConc.astype(np.float)), IL15_data[(m * 4): ((m + 1) * 4)])
        axis = axis + 1


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
