"""
This file is responsible for performing calculations that allow us to compare our fitting results with the Ring paper in figure1.py
"""
import numpy as np
from .model import getTotalActiveSpecies, runCkineUP, getSurfaceIL2RbSpecies, nParams, getSurfaceGCSpecies


class surf_IL2Rb:
    '''Generate values to match the surface IL2Rb measurements used in fitting'''

    def __init__(self):
        # import function returns from model.py
        self.IL2Rb_species_IDX = getSurfaceIL2RbSpecies()

    def parallelCalc(self, unkVec, cytokine, conc, t):
        """ Calculates the surface IL2Rb over time in parallel for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine, :] = conc
        unkVec = np.transpose(unkVec).copy()  # transpose the matrix (save view as a new copy)
        returnn, retVal = runCkineUP(t, unkVec)
        assert retVal >= 0
        return np.dot(returnn, self.IL2Rb_species_IDX)

    def calc(self, unkVec, t):
        '''This function uses an unkVec that has the same elements as the unkVec in fit.py'''
        assert unkVec.shape[0] == nParams()
        N = len(t)
        K = unkVec.shape[1]

        # set IL2 concentrations
        unkVecIL2RaMinus = unkVec.copy()
        unkVecIL2RaMinus[22, :] = np.zeros((unkVec.shape[1]))

        # calculate IL2 stimulation
        a = self.parallelCalc(unkVec, 0, 1., t).reshape((K, N))
        b = self.parallelCalc(unkVec, 0, 500., t).reshape((K, N))
        c = self.parallelCalc(unkVecIL2RaMinus, 0, 1., t).reshape((K, N))
        d = self.parallelCalc(unkVecIL2RaMinus, 0, 500., t).reshape((K, N))

        # calculate IL15 stimulation
        e = self.parallelCalc(unkVec, 1, 1., t).reshape((K, N))
        f = self.parallelCalc(unkVec, 1, 500., t).reshape((K, N))
        g = self.parallelCalc(unkVecIL2RaMinus, 1, 1., t).reshape((K, N))
        h = self.parallelCalc(unkVecIL2RaMinus, 1, 500., t).reshape((K, N))

        catVec = np.concatenate((a, b, c, d, e, f, g, h), axis=1)
        for ii in range(K):
            catVec[ii] = catVec[ii] / a[ii, 0]  # normalize by a[0] for each row
        return catVec


class pstat:
    '''Generate values to match the pSTAT5 measurements used in fitting'''

    def __init__(self):
        # import function returns from model.py
        self.activity = getTotalActiveSpecies().astype(np.float64)
        self.ts = np.array([500.])  # was 500. in literature

    def parallelCalc(self, unkVec, cytokine, conc):
        """ Calculates the pSTAT activities in parallel for a 2-D array of unkVec. """
        unkVec = unkVec.copy()
        unkVec[cytokine, :] = conc
        unkVec = np.transpose(unkVec).copy()  # transpose the matrix (save view as a new copy)
        returnn, retVal = runCkineUP(self.ts, unkVec)
        assert retVal >= 0
        return np.dot(returnn, self.activity)

    def calc(self, unkVec, scale, cytokC):
        '''This function uses an unkVec that has the same elements as the unkVec in fit.py'''
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
            actVec_IL2[:, x] = self.parallelCalc(unkVec, 0, conc)
            actVec_IL2_IL2Raminus[:, x] = self.parallelCalc(unkVec_IL2Raminus, 0, conc)
            actVec_IL15[:, x] = self.parallelCalc(unkVec, 1, conc)
            actVec_IL15_IL2Raminus[:, x] = self.parallelCalc(unkVec_IL2Raminus, 1, conc)

        # put together into one vector & normalize by scale
        actVec = np.concatenate((actVec_IL2, actVec_IL2_IL2Raminus, actVec_IL15, actVec_IL15_IL2Raminus), axis=1)
        actVec = actVec / (actVec + scale)

        for ii in range(K):
            actVec[ii] = actVec[ii] / np.max(actVec[ii])  # normalize by the max value of each row

        return actVec


class surf_gc:
    """ This class is responsible for calculating the percent of gamma chain on the cell surface. The experimental conditions match those of the surface IL2Rb measurements in Ring et al. """

    def __init__(self):
        # import function returns from model.py
        self.gc_species_IDX = getSurfaceGCSpecies()

    def parallelCalc(self, unkVec, cytokine, conc, t):
        """ Calculates the surface gc over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine, :] = conc
        unkVec = np.transpose(unkVec).copy()  # transpose the matrix (save view as a new copy)
        returnn, retVal = runCkineUP(t, unkVec)
        assert retVal >= 0
        return np.dot(returnn, self.gc_species_IDX)

    def calc(self, unkVec, t):
        '''This function calls single Calc for all the experimental combinations of interest; it uses an unkVec that has the same elements as the unkVec in fit.py'''
        assert unkVec.shape[0] == nParams()
        N = len(t)
        K = unkVec.shape[1]

        # set IL2 concentrations
        unkVecIL2RaMinus = unkVec.copy()
        unkVecIL2RaMinus[22, :] = np.zeros((K))

        # calculate IL2 stimulation
        a = self.parallelCalc(unkVecIL2RaMinus, 0, 1000., t).reshape((K, N))

        for ii in range(K):
            a[ii] = a[ii] / a[ii, 0]  # normalize by a[0] for each row
        return a
