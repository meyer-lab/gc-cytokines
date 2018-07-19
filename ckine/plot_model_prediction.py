import numpy as np
import matplotlib.pyplot as plt
from .model import getTotalActiveSpecies, runCkineU, getSurfaceIL2RbSpecies, nSpecies, nParams, getSurfaceGCSpecies


class surf_IL2Rb:
    '''Generate values to match the surface IL2Rb measurements used in fitting'''
    def __init__(self):
        # import function returns from model.py
        self.IL2Rb_species_IDX = getSurfaceIL2RbSpecies()

    def singleCalc(self, unkVec, cytokine, conc, t):
        """ Calculates the surface IL2Rb over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc

        returnn, retVal = runCkineU(t, unkVec)

        assert retVal >= 0

        a = np.dot(returnn, self.IL2Rb_species_IDX)

        return a

    def calc(self, unkVec, t):
        '''This function uses an unkVec that has the same elements as the unkVec in fit.py'''

        assert unkVec.size == nParams()

        # set IL2 concentrations
        unkVecIL2RaMinus = unkVec.copy()
        unkVecIL2RaMinus[22] = 0.

        # calculate IL2 stimulation
        a = self.singleCalc(unkVec, 0, 1., t)
        b = self.singleCalc(unkVec, 0, 500., t)
        c = self.singleCalc(unkVecIL2RaMinus, 0, 1., t)
        d = self.singleCalc(unkVecIL2RaMinus, 0, 500., t)

        # calculate IL15 stimulation
        e = self.singleCalc(unkVec, 1, 1., t)
        f = self.singleCalc(unkVec, 1, 500., t)
        g = self.singleCalc(unkVecIL2RaMinus, 1, 1., t)
        h = self.singleCalc(unkVecIL2RaMinus, 1, 500., t)

        return (np.concatenate((a, b, c, d, e, f, g, h)) / a[0])

class pstat:
    '''Generate values to match the pSTAT5 measurements used in fitting'''
    def __init__(self):
        # import function returns from model.py
        self.activity = getTotalActiveSpecies().astype(np.float64)
        
        self.ts = np.array([500.]) # was 500. in literature

    def singleCalc(self, unkVec, cytokine, conc):
        """ Calculates the surface IL2Rb over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc

        returnn, retVal = runCkineU(self.ts, unkVec)

        assert retVal >= 0

        return np.dot(returnn, self.activity)

    def calc(self, unkVec, cytokC):
        '''This function uses an unkVec that has the same elements as the unkVec in fit.py'''
        assert unkVec.size == nParams()

        unkVec_IL2Raminus = unkVec.copy()
        unkVec_IL2Raminus[22] = 0.0 # set IL2Ra expression rate to 0

        # Calculate activities
        actVec_IL2 = np.fromiter((self.singleCalc(unkVec, 0, x) for x in cytokC), np.float64)
        actVec_IL2_IL2Raminus = np.fromiter((self.singleCalc(unkVec_IL2Raminus, 0, x) for x in cytokC), np.float64)
        actVec_IL15 = np.fromiter((self.singleCalc(unkVec, 1, x) for x in cytokC), np.float64)
        actVec_IL15_IL2Raminus = np.fromiter((self.singleCalc(unkVec_IL2Raminus, 1, x) for x in cytokC), np.float64)

        # Normalize to the maximal activity, put together into one vector
        actVec = np.concatenate((actVec_IL2, actVec_IL2_IL2Raminus, actVec_IL15, actVec_IL15_IL2Raminus))

        return (actVec / np.max(actVec))

class surf_gc:
    def __init__(self):
        # import function returns from model.py
        self.gc_species_IDX = getSurfaceGCSpecies()
        
    def singleCalc(self, unkVec, cytokine, conc, t):
        """ Calculates the surface gc over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc

        returnn, retVal = runCkineU(t, unkVec)

        assert retVal >= 0

        a = np.dot(returnn, self.gc_species_IDX)

        return a
    
    def calc(self, unkVec, t):
        '''This function calls single Calc for all the experimental combinations of interest; it uses an unkVec that has the same elements as the unkVec in fit.py'''

        assert unkVec.size == nParams()

        # set IL2 concentrations
        unkVecIL2RaMinus = unkVec.copy()
        unkVecIL2RaMinus[22] = 0.

        # calculate IL2 stimulation
        a = self.singleCalc(unkVec, 0, 1., t)
        b = self.singleCalc(unkVec, 0, 500., t)
        c = self.singleCalc(unkVecIL2RaMinus, 0, 1., t)
        d = self.singleCalc(unkVecIL2RaMinus, 0, 500., t)

        # calculate IL15 stimulation
        e = self.singleCalc(unkVec, 1, 1., t)
        f = self.singleCalc(unkVec, 1, 500., t)
        g = self.singleCalc(unkVecIL2RaMinus, 1, 1., t)
        h = self.singleCalc(unkVecIL2RaMinus, 1, 500., t)

        return (np.concatenate((a, b, c, d, e, f, g, h)) / a[0])
    