"""
This creates Figure S2. Full panel of Geweke convergence tests.
"""
import string
import pymc3 as pm
import matplotlib.cm as cm
import numpy as np
from .figureCommon import subplotLabel, getSetup, traf_names
from ..imports import import_samples_2_15, import_samples_4_7


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 6), (3, 4))

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    plot_geweke_2_15(ax[0:4], True)
    plot_geweke_2_15(ax[4:7], False)
    plot_geweke_4_7(ax[8:11])

    f.tight_layout()

    return f


def plot_geweke_2_15(ax, traf):
    """ Uses geweke criterion to evaluate model convergence during fitting. """
    trace = import_samples_2_15(Traf=traf, ret_trace=True)  # return the trace

    # use use trace to calculate geweke z-scores
    score = pm.diagnostics.geweke(trace, first=0.1, last=0.5, intervals=20)

    # plot the scores for rxn rates
    rxn_len = len(score[0]['rxn'])
    rxn_names = [r'$k_{4}$', r'$k_{5}$', r'$k_{16}$', r'$k_{17}$', r'$k_{22}$', r'$k_{23}$']
    colors = cm.rainbow(np.linspace(0, 1, rxn_len))
    for ii in range(rxn_len):
        ax[0].scatter(score[0]['rxn'][ii][:, 0], score[0]['rxn'][ii][:, 1], marker='o', s=25, color=colors[ii], label=rxn_names[ii])  # plot all rates within rxn
    ax[0].axhline(-1., c='r')
    ax[0].axhline(1., c='r')
    ax[0].set(ylim=(-1.25, 1.25), xlim=(0 - 10, .5 * trace['rxn'].shape[0] / 2 + 10),
              xlabel="Position in Chain", ylabel="Geweke Score")
    if traf:
        ax[0].set_title('IL-2/-15 traf model: reverse rxn')
    else:
        ax[0].set_title('IL-2/-15 no traf model: reverse rxn')
    ax[0].legend()

    # plot the scores for receptor expression rates
    rexpr_names = ['IL-2Rα', 'IL-2Rβ', 'IL-15Rα']
    rexpr_len = len(rexpr_names)
    colors = cm.rainbow(np.linspace(0, 1, rexpr_len))

    # plot IL2Ra, IL2Rb, and gc
    ax[1].scatter(score[0]['Rexpr_2Ra'][:, 0], score[0]['Rexpr_2Ra'][:, 1], marker='o', s=25, color=colors[0], label=rexpr_names[0])
    ax[1].scatter(score[0]['Rexpr_2Rb'][:, 0], score[0]['Rexpr_2Rb'][:, 1], marker='o', s=25, color=colors[1], label=rexpr_names[1])
    ax[1].scatter(score[0]['Rexpr_15Ra'][:, 0], score[0]['Rexpr_15Ra'][:, 1], marker='o', s=25, color=colors[2], label=rexpr_names[2])
    ax[1].axhline(-1., c='r')
    ax[1].axhline(1., c='r')
    ax[1].set(ylim=(-1.25, 1.25), xlim=(0 - 10, .5 * trace['Rexpr_2Ra'].shape[0] / 2 + 10),
              xlabel="Position in Chain", ylabel="Geweke Score")
    if traf:
        ax[1].set_title('IL-2/-15 traf model: receptor expression')
    else:
        ax[1].set_title('IL-2/-15 no-traf model: receptor abundance')
    ax[1].legend()

    # plot the scores for scaling constant and kfwd
    ax[2].scatter(score[0]['scales'][:, 0], score[0]['scales'][:, 1], marker='o', s=25, color='g', label=r'$C_{5}$')
    ax[2].scatter(score[0]['kfwd'][:, 0], score[0]['kfwd'][:, 1], marker='o', s=25, color='b', label=r'$k_{fwd}$')
    ax[2].axhline(-1., c='r')
    ax[2].axhline(1., c='r')
    ax[2].set(ylim=(-1.25, 1.25), xlim=(0 - 10, .5 * trace['kfwd'].shape[0] / 2 + 10),
              xlabel="Position in Chain", ylabel="Geweke Score")
    if traf:
        ax[2].set_title(r'IL-2/-15 traf model: $C_{5}$ and $k_{fwd}$')
    else:
        ax[2].set_title(r'IL-2/-15 no-traf model: $C_{5}$ and $k_{fwd}$')
    ax[2].legend()

    if traf is True:
        colors = cm.rainbow(np.linspace(0, 1, 5))
        tr_names = traf_names()
        ax[3].scatter(score[0]['endo'][:, 0], score[0]['endo'][:, 1], marker='o', s=25, color=colors[0], label=tr_names[0])
        ax[3].scatter(score[0]['activeEndo'][:, 0], score[0]['activeEndo'][:, 1], marker='o', s=25, color=colors[1], label=tr_names[1])
        ax[3].scatter(score[0]['sortF'][:, 0], score[0]['sortF'][:, 1], marker='o', s=25, color=colors[2], label=r'$f_{sort}$')  # sortF not in traf_names()
        ax[3].scatter(score[0]['kRec'][:, 0], score[0]['kRec'][:, 1], marker='o', s=25, color=colors[3], label=tr_names[2])
        ax[3].scatter(score[0]['kDeg'][:, 0], score[0]['kDeg'][:, 1], marker='o', s=25, color=colors[4], label=tr_names[3])
        ax[3].axhline(-1., c='r')
        ax[3].axhline(1., c='r')
        ax[3].set(ylim=(-1.25, 1.25), xlim=(0 - 10, .5 * trace['endo'].shape[0] / 2 + 10),
                  xlabel="Position in Chain", ylabel="Geweke Score", title="IL-2/-15 traf model: traf rates")
        ax[3].legend()


def plot_geweke_4_7(ax):
    """ Generating Geweke plots using the traces from IL-4 and IL-7 fitting to Gonnord data. """
    trace = import_samples_4_7(ret_trace=True)  # return the trace

    # use use trace to calculate geweke z-scores
    score = pm.diagnostics.geweke(trace, first=0.1, last=0.5, intervals=20)

    # plot the scores for rxn rates
    colors = cm.rainbow(np.linspace(0, 1, 2))

    ax[0].scatter(score[0]['k27rev'][:, 0], score[0]['k27rev'][:, 1], marker='o', s=25, color=colors[0], label=r'$k_{27}$')
    ax[0].scatter(score[0]['k33rev'][:, 0], score[0]['k33rev'][:, 1], marker='o', s=25, color=colors[1], label=r'$k_{33}$')
    ax[0].axhline(-1., c='r')
    ax[0].axhline(1., c='r')
    ax[0].set(ylim=(-1.25, 1.25), xlim=(0 - 10, .5 * trace['k27rev'].shape[0] / 2 + 10),
              xlabel="Position in Chain", ylabel="Geweke Score")
    ax[0].set_title('IL-4/-7 model: reverse rxn')
    ax[0].legend()

    # plot the scores for scaling constant and kfwd
    ax[1].scatter(score[0]['scales'][0][:, 0], score[0]['scales'][0][:, 1], marker='o', s=25, color='g', label=r'$C_{5}$')
    ax[1].scatter(score[0]['scales'][1][:, 0], score[0]['scales'][1][:, 1], marker='o', s=25, color='c', label=r'$C_{6}$')
    ax[1].scatter(score[0]['kfwd'][:, 0], score[0]['kfwd'][:, 1], marker='o', s=25, color='b', label=r'$k_{fwd}$')
    ax[1].axhline(-1., c='r')
    ax[1].axhline(1., c='r')
    ax[1].set(ylim=(-1.25, 1.25), xlim=(0 - 10, .5 * trace['kfwd'].shape[0] / 2 + 10),
              xlabel="Position in Chain", ylabel="Geweke Score")
    ax[1].set_title(r'IL-4/-7 model: $C_{5}$, $C_{6}$, and $k_{fwd}$')
    ax[1].legend()

    colors = cm.rainbow(np.linspace(0, 1, 5))
    tr_names = traf_names()
    ax[2].scatter(score[0]['endo'][:, 0], score[0]['endo'][:, 1], marker='o', s=25, color=colors[0], label=tr_names[0])
    ax[2].scatter(score[0]['activeEndo'][:, 0], score[0]['activeEndo'][:, 1], marker='o', s=25, color=colors[1], label=tr_names[1])
    ax[2].scatter(score[0]['sortF'][:, 0], score[0]['sortF'][:, 1], marker='o', s=25, color=colors[2], label=r'$f_{sort}$')  # sortF not in traf_names()
    ax[2].scatter(score[0]['kRec'][:, 0], score[0]['kRec'][:, 1], marker='o', s=25, color=colors[3], label=tr_names[2])
    ax[2].scatter(score[0]['kDeg'][:, 0], score[0]['kDeg'][:, 1], marker='o', s=25, color=colors[4], label=tr_names[3])
    ax[2].axhline(-1., c='r')
    ax[2].axhline(1., c='r')
    ax[2].set(ylim=(-1.25, 1.25), xlim=(0 - 10, .5 * trace['endo'].shape[0] / 2 + 10),
              xlabel="Position in Chain", ylabel="Geweke Score", title="IL-4/-7 model: traf rates")
    ax[2].legend()
