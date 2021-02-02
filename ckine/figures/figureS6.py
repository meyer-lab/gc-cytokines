"""
This creates Figure 4. Comparison of Experimental verus Predicted Activity across IL2 and IL15 concentrations.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from .figureCommon import subplotLabel, getSetup, catplot_comparison, nllsq_EC50, global_legend
from .figure4 import plot_corrcoef, WT_EC50s
from .figureS5 import plot_exp_v_pred
from ..imports import import_pstat, import_Rexpr, import_samples_2_15

unkVec_2_15_glob = import_samples_2_15(Traf=False, N=1)  # use one rate


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3), multz={1: 1})
    subplotLabel(ax)

    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    tpsSC = np.array([0.5, 1.0]) * 60.0
    df = WT_EC50s(unkVec_2_15_glob, Traf=False)
    catplot_comparison(ax[0], df, Mut=False)  # compare experiments to model predictions
    plot_corrcoef(ax[1], tps, unkVec_2_15_glob, Traf=False)  # find correlation coefficients
    global_legend(ax[1], Mut=True, exppred=False)  # add legend subplots A-C

    plot_exp_v_pred(ax[2:8], tpsSC, cell_subset=["NK", "CD8+", "T-reg"], Traf=False)  # NK, CD8+, and Treg subplots taken from fig S5

    return f
