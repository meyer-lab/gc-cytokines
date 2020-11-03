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


class pstat:
    """Generate values to match the pSTAT5 measurements used in fitting"""

    def __init__(self):
        # import function returns from model.py
        self.activity = getTotalActiveSpecies().astype(np.float64)
        self.ts = np.array([500.0])  # was 500. in literature

    def calc(self, unkVec, cytokC):
        """This function uses an unkVec that has the same elements as the unkVec in fit.py"""
        assert unkVec.shape[0] == nParams()
        K = unkVec.shape[1]

        unkVec_IL2Raminus = unkVec.copy()
        unkVec_IL2Raminus[22, :] = np.zeros((K))  # set IL2Ra expression rates to 0

        actVec_IL2 = np.zeros((K, len(cytokC)))
        actVec_IL2_IL2Raminus = actVec_IL2.copy()
        actVec_IL15 = actVec_IL2.copy()
        actVec_IL15_IL2Raminus = actVec_IL2.copy()

        # Calculate activities
        for x, conc in enumerate(cytokC):
            actVec_IL2[:, x] = parallelCalc(unkVec, 0, conc, self.ts, self.activity, reshapeP=False)
            actVec_IL2_IL2Raminus[:, x] = parallelCalc(unkVec_IL2Raminus, 0, conc, self.ts, self.activity, reshapeP=False)
            actVec_IL15[:, x] = parallelCalc(unkVec, 1, conc, self.ts, self.activity, reshapeP=False)
            actVec_IL15_IL2Raminus[:, x] = parallelCalc(unkVec_IL2Raminus, 1, conc, self.ts, self.activity, reshapeP=False)

        # put together into one vector
        actVec = np.concatenate((actVec_IL2, actVec_IL2_IL2Raminus, actVec_IL15, actVec_IL15_IL2Raminus), axis=1)

        return actVec / actVec.max(axis=1, keepdims=True)  # normalize by the max value of each row


class surf_gc:
    """ This class is responsible for calculating the percent of gamma chain on the cell surface. The experimental conditions match those of the surface IL2Rb measurements in Ring et al. """

    def __init__(self):
        # import function returns from model.py
        self.gc_species_IDX = getSurfaceGCSpecies()

    def calc(self, unkVec, t):
        """This function calls single Calc for all the experimental combinations of interest; it uses an unkVec that has the same elements as the unkVec in fit.py"""
        assert unkVec.shape[0] == nParams()
        K = unkVec.shape[1]

        # set IL2 concentrations
        unkVecIL2RaMinus = unkVec.copy()
        unkVecIL2RaMinus[22, :] = 0.0

        # calculate IL2 stimulation
        a = parallelCalc(unkVecIL2RaMinus, 0, 1000.0, t, self.gc_species_IDX)

        for ii in range(K):
            a[ii] = a[ii] / a[ii, 0]  # normalize by a[0] for each row
        return a
