"""
This creates Figure S5.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Get a list of the axis objects and create the figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax[0], 'A')

    return f
