"""
This creates Figure 4.
"""
from .figureCommon import subplotLabel, getSetup
from ..simulation_plotting import solve_IL2_IL15


def makeFigure():
    """ Build the figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    subplotLabel(ax[0], 'A')

    nonadditive_activity(ax[0], 1.0, 50, 50.0, '1nM')
    nonadditive_activity(ax[1], 500.0, 50, 240.0, '500nM')

    return f


def nonadditive_activity(ax, final_conc, num, time, conc):
    """ Plots IL2 & IL15 activity versus concentration for a given time point. """
    active_matrix, _, xaxis = solve_IL2_IL15(final_conc, num, time)
    ax.plot(xaxis, active_matrix[0], 'b', label='IL2 Activity')
    ax.plot(xaxis, active_matrix[1], 'r', label='IL15 Activity')
    ax.plot(xaxis, active_matrix[2], 'k', label='Total Activity')
    ax.set(xlabel='Log(IL2/IL15)', ylabel='Activity', title='Non-additive effects between IL2/15 in YT-1 cells (' + conc + ')')
    ax.set_ylim(bottom=0)
    ax.legend()
