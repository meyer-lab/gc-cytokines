"""
This creates Figure S1.
"""
import string
from .figureCommon import subplotLabel, getSetup, import_samples_2_15, kfwd_info
from .figure1 import pstat_act, violinPlots, rateComp

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (2, 3), mults=[1], multz={1:2}, empts=[3])

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    unkVec, scales = import_samples_2_15(Fig1=False)
    kfwd_avg, kfwd_std = kfwd_info(unkVec)
    print("kfwd = " + str(kfwd_avg) + " +/- " + str(kfwd_std))
    pstat_act(ax[0], unkVec, scales, Fig1=False)
    rateComp(ax[1], unkVec)
    violinPlots(ax[2:4], unkVec, scales, Fig1=False)


    f.tight_layout()

    return f
