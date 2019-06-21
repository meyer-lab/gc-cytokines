"""
This creates Figure S1.
"""
import string
from .figureCommon import subplotLabel, getSetup, kfwd_info
from .figure1 import pstat_act, violinPlots, rateComp, legend_2_15
from ..imports import import_samples_2_15


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 4), (2, 3), multz={1: 1})

    # add legend
    leg_ind = 2
    legend_2_15(ax[leg_ind], location="center")

    for ii, item in enumerate(ax):
        # add conditionals to skip the legend
        if ii < leg_ind:
            subplotLabel(item, string.ascii_uppercase[ii])
        elif ii > leg_ind:
            subplotLabel(item, string.ascii_uppercase[ii - 1])

    unkVec, scales = import_samples_2_15(Traf=False, N=100)
    full_unkVec, full_scales = import_samples_2_15(Traf=False)
    kfwd_avg, kfwd_std = kfwd_info(full_unkVec)
    print("kfwd = " + str(kfwd_avg) + " +/- " + str(kfwd_std))
    pstat_act(ax[0], unkVec, scales)
    rateComp(ax[1], full_unkVec)
    violinPlots(ax[3:5], full_unkVec, full_scales, Traf=False)

    return f
