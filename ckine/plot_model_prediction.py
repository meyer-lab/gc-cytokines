"""
This file is responsible for performing calculations that allow us to compare our fitting results with the Ring paper in figure1.py
"""
import numpy as np
from .model import getTotalActiveSpecies, runCkineUP, nParams, getSurfaceGCSpecies


def parallelCalc(unkVec, cytokine, conc, t, condense, reshapeP=True):
    """ Calculates the species over time in parallel for one condition. """
    unkVec = np.transpose(unkVec).copy()
    unkVec[:, cytokine] = conc
    outt = np.dot(runCkineUP(t, unkVec), condense)

    if reshapeP is True:
        return outt.reshape((unkVec.shape[0], len(t)))
    return outt
