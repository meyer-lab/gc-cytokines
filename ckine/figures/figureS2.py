"""
This creates Figure S2. Full panel of Geweke convergence tests.
"""
import string
import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
from arviz.stats.diagnostics import geweke
from .figureCommon import subplotLabel, getSetup
from ..imports import import_samples_2_15, import_samples_4_7


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 1))

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    plot_geweke_2_15(ax[0], True)
    plot_geweke_2_15(ax[1], False)
    plot_geweke_4_7(ax[2])

    return f


def plot_geweke_2_15(ax, traf):
    """ Uses geweke criterion to evaluate model convergence during fitting. """
    trace = import_samples_2_15(Traf=traf, ret_trace=True)  # return the trace

    df = pd.DataFrame()
    df[r'$k_{4}$'] = geweke(trace["rxn__0"])
    df[r'$k_{5}$'] = geweke(trace["rxn__1"])
    df[r'$k_{16}$'] = geweke(trace["rxn__2"])
    df[r'$k_{17}$'] = geweke(trace["rxn__3"])
    df[r'$k_{22}$'] = geweke(trace["rxn__4"])
    df[r'$k_{23}$'] = geweke(trace["rxn__5"])

    df['IL-2Rα'] = geweke(trace["Rexpr_2Ra__0"])
    df['IL-2Rβ'] = geweke(trace["Rexpr_2Rb__0"])
    df['IL-15Rα'] = geweke(trace["Rexpr_15Ra__0"])
    df[r'$γ_{c}$'] = geweke(trace["Rexpr_gc__0"])
    df[r'$C_{5}$'] = geweke(trace["scales__0"])
    df[r'$k_{fwd}$'] = geweke(trace["kfwd__0"])

    if traf:  # add the trafficking parameters if necessary & set proper title
        df[r'$k_{endo}$'] = geweke(trace["endo__0"])
        df[r'$k_{endo,a}$'] = geweke(trace["activeEndo__0"])
        df[r'$f_{sort}$'] = geweke(trace["sortF__0"])
        df[r'$k_{rec}$'] = geweke(trace["kRec__0"])
        df[r'$k_{deg}$'] = geweke(trace["kDeg__0"])
        ax.set_title(r'IL-2/-15 trafficking model')
    else:
        ax.set_title(r'IL-2/-15 no trafficking model')

    sns.scatterplot(data=df, ax=ax)

    ax.set_xticklabels(df.columns,  # use keys from dict as x-axis labels
                       rotation=25,
                       rotation_mode="anchor",
                       ha="right",
                       fontsize=8,
                       position=(0, 0.075))
    ax.get_legend().set_visible(False)  # remove legend created by sns
    ax.axhline(1., c='r')  # line to denote acceptable threshold of standard deviations
    ax.set(ylim=(-0.1, 1.25), ylabel=r"max |z-score|")


def plot_geweke_4_7(ax, traf=True):
    """ Generating Geweke plots using the traces from IL-4 and IL-7 fitting to Gonnord data. """
    trace = import_samples_4_7(ret_trace=True)  # return the trace

    df = pd.DataFrame()
    df[r'$k_{27}$'] = geweke(trace["k27rev__0"])
    df[r'$k_{33}$'] = geweke(trace["k33rev__0"])
    df[r'$C_{5}$'] = geweke(trace["scales__0"])
    df[r'$C_{6}$'] = geweke(trace["scales__1"])
    df[r'$k_{fwd}$'] = geweke(trace["kfwd__0"])

    if traf:  # add the trafficking parameters if necessary & set proper title
        df[r'$k_{endo}$'] = geweke(trace["endo__0"])
        df[r'$k_{endo,a}$'] = geweke(trace["activeEndo__0"])
        df[r'$f_{sort}$'] = geweke(trace["sortF__0"])
        df[r'$k_{rec}$'] = geweke(trace["kRec__0"])
        df[r'$k_{deg}$'] = geweke(trace["kDeg__0"])
        ax.set_title(r'IL-4/-7 trafficking model')
    else:
        ax.set_title(r'IL-4/-7 no trafficking model')

    sns.scatterplot(data=df, ax=ax)

    ax.set_xticklabels(df.columns,  # use keys from dict as x-axis labels
                       rotation=25,
                       rotation_mode="anchor",
                       ha="right",
                       fontsize=8,
                       position=(0, 0.075))
    ax.get_legend().set_visible(False)  # remove legend created by sns
    ax.axhline(1., c='r')  # line to denote acceptable threshold of standard deviations
    ax.set(ylim=(-0.1, 1.25), ylabel=r"max |z-score|")
