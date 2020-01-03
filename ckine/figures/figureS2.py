"""
This creates Figure S2. Full panel of Geweke convergence tests.
"""
import string
import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
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

    # use use trace to calculate geweke z-scores
    score = pm.diagnostics.geweke(trace, first=0.1, last=0.5, intervals=20)

    # take z-score from interval with maximum absolute value (i.e. the interval that converged the worst)
    dictt = {r'$k_{4}$': max(abs(score[0]['rxn'][0][:, 1])),
             r'$k_{5}$': max(abs(score[0]['rxn'][1][:, 1])),
             r'$k_{16}$': max(abs(score[0]['rxn'][2][:, 1])),
             r'$k_{17}$': max(abs(score[0]['rxn'][3][:, 1])),
             r'$k_{22}$': max(abs(score[0]['rxn'][4][:, 1])),
             r'$k_{23}$': max(abs(score[0]['rxn'][5][:, 1])),
             'IL-2Rα': max(abs(score[0]['Rexpr_2Ra'][:, 1])),
             'IL-2Rβ': max(abs(score[0]['Rexpr_2Rb'][:, 1])),
             'IL-15Rα': max(abs(score[0]['Rexpr_15Ra'][:, 1])),
             r'$γ_{c}$': max(abs(score[0]['Rexpr_gc'][:, 1])),
             r'$C_{5}$': max(abs(score[0]['scales'][:, 1])),
             r'$k_{fwd}$': max(abs(score[0]['kfwd'][:, 1]))}

    if traf:  # add the trafficking parameters if necessary & set proper title
        dictt.update({r'$k_{endo}$': max(abs(score[0]['endo'][:, 1])),
                      r'$k_{endo,a}$': max(abs(score[0]['activeEndo'][:, 1])),
                      r'$f_{sort}$': max(abs(score[0]['sortF'][:, 1])),
                      r'$k_{rec}$': max(abs(score[0]['kRec'][:, 1])),
                      r'$k_{deg}$': max(abs(score[0]['kDeg'][:, 1]))})
        ax.set_title(r'IL-2/-15 trafficking model')
    else:
        ax.set_title(r'IL-2/-15 no trafficking model')

    df = pd.DataFrame.from_dict(dictt, orient='index')
    sns.scatterplot(data=np.abs(df), ax=ax)

    ax.set_xticklabels(list(dictt.keys()),  # use keys from dict as x-axis labels
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

    # use use trace to calculate geweke z-scores
    score = pm.diagnostics.geweke(trace, first=0.1, last=0.5, intervals=20)

    # take score from the 10th interval in the chain... can change this to a random int later
    dictt = {r'$k_{27}$': max(abs(score[0]['k27rev'][:, 1])),
             r'$k_{33}$': max(abs(score[0]['k33rev'][:, 1])),
             r'$C_{5}$': max(abs(score[0]['scales'][0][:, 1])),
             r'$C_{6}$': max(abs(score[0]['scales'][1][:, 1])),
             r'$k_{fwd}$': max(abs(score[0]['kfwd'][:, 1]))}

    if traf:  # add the trafficking parameters if necessary & set proper title
        dictt.update({r'$k_{endo}$': max(abs(score[0]['endo'][:, 1])),
                      r'$k_{endo,a}$': max(abs(score[0]['activeEndo'][:, 1])),
                      r'$f_{sort}$': max(abs(score[0]['sortF'][:, 1])),
                      r'$k_{rec}$': max(abs(score[0]['kRec'][:, 1])),
                      r'$k_{deg}$': max(abs(score[0]['kDeg'][:, 1]))})
        ax.set_title(r'IL-4/-7 trafficking model')
    else:
        ax.set_title(r'IL-4/-7 no trafficking model')

    df = pd.DataFrame.from_dict(dictt, orient='index')
    sns.scatterplot(data=np.abs(df), ax=ax)

    ax.set_xticklabels(list(dictt.keys()),  # use keys from dict as x-axis labels
                       rotation=25,
                       rotation_mode="anchor",
                       ha="right",
                       fontsize=8,
                       position=(0, 0.075))
    ax.get_legend().set_visible(False)  # remove legend created by sns
    ax.axhline(1., c='r')  # line to denote acceptable threshold of standard deviations
    ax.set(ylim=(-0.1, 1.25), ylabel=r"max |z-score|")
