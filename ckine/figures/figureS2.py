"""
This creates Figure S2.
"""
import string
import os
import pickle
import numpy as np
import pandas as pds
from .figureCommon import subplotLabel, getSetup, plot_timepoint, plot_cells, plot_ligands, plot_values
from ..Tensor_analysis import reorient_factors

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    x, y = 7, 4
    ssize = 3
    ax, f = getSetup((ssize*y, ssize*x), (x, y))
    return f
