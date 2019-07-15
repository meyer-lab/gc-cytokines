"""
This creates Figure 4.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Build the figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    subplotLabel(ax[0], 'A')

    return f
