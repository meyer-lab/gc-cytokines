"""
This creates Figure S1.
"""
from .figureCommon import subplotLabel, getSetup, legend_2_15
from .figure1 import pstat_act, violinPlots, rateComp
from ..imports import import_samples_2_15


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 6), (2, 3), multz={1: 1})

    # add legend
    leg_ind = 2
    legend_2_15(ax[leg_ind], location="center")

    # Add subplot labels
    axLabel = ax.copy()
    del axLabel[leg_ind]
    subplotLabel(axLabel, hstretch={1: 2.35})

    unkVec = import_samples_2_15(Traf=False, N=100)
    full_unkVec = import_samples_2_15(Traf=False)
    pstat_act(ax[0], unkVec)
    rateComp(ax[1], full_unkVec, fsize=8)
    violinPlots(ax[3:5], full_unkVec, Traf=False)

    return f
