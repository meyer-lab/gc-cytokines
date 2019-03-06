"""
This creates Figure S1.
"""
import string
from .figureCommon import subplotLabel, getSetup, traf_names, plot_conf_int, import_samples_2_15, kfwd_info
from .figure1 import pstat_act, violinPlots, rateComp

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (2, 3), mults=[3], multz={3:2}, empts=[2, 5])

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    unkVec, scales = import_samples_2_15(Fig1 = False)
    pstat_act(ax[1], unkVec, scales, Fig1 = False)

    violinPlots(ax[0], unkVec, Fig1 = False)
    rateComp(ax[2], unkVec)


    f.tight_layout()

    return f
