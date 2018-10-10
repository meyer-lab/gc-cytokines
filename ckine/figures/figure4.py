"""
This creates Figure 4.
"""
import string
import pandas as pd
import seaborn as sns
import numpy as np
from .figureCommon import subplotLabel, getSetup, import_samples_2_15, import_samples_4_7

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4), mults=[0], multz={0: 2})

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    f.tight_layout()

    unkVec_2_15 = import_samples_2_15()
    unkVec_4_7, scales = import_samples_4_7()
    relativeGC(ax[0], unkVec_2_15, unkVec_4_7)

    return f


def relativeGC(ax, unkVec2, unkVec4):
    """ This function compares the relative complex affinities for GC. The rates included in this violing plot will be k4rev, k10rev, k17rev, k22rev, k27rev, and k33rev. We're currently ignoring k31rev (IL9) and k35rev (IL21) since we don't fit to any of its data. """

    # assign values from unkVec
    k4rev, k5rev, k16rev, k17rev, k22rev, k27rev, k33rev = unkVec2[7, :], unkVec2[8, :], unkVec2[9, :], unkVec2[10, :], unkVec2[11, :], unkVec4[13, :], unkVec4[15, :]

    # back-out k10 with ratio
    k10rev = 12.0 * k5rev / 1.5 # doi:10.1016/j.jmb.2004.04.038

    # add each rate duo as separate column in dataframe
    df = pd.DataFrame({'2-2Ra': k4rev, '2-2Rb': k5rev, '2-2Ra-2Rb': k10rev, '15-15Ra': k16rev, '15-2Rb': k17rev, '15-15Ra-2Rb': k22rev, '7-7Ra': k27rev, '4-4Ra': k33rev})

    col_list = ["violet", "violet", "violet", "goldenrod", "goldenrod", "goldenrod", "blue", "lightblue"]
    col_list_palette = sns.xkcd_palette(col_list)
    cmap = sns.set_palette(col_list_palette)

    a = sns.violinplot(data=np.log10(df), ax=ax, linewidth=0, bw=10, cmap=cmap, scale='width')
    a.set_xticklabels(a.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.075))
    a.set(title="Relative gc affinity", ylabel="log10 of value")
    