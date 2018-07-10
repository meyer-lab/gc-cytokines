import numpy as np
import matplotlib.pyplot as plt
from .model import getTotalActiveSpecies, runCkineU, getSurfaceIL2RbSpecies, nSpecies, nParams


class surf_IL2Rb:
    '''Generate values to match the surface IL2Rb measurements used in fitting'''
    def __init__(self):
        # times from experiment are hard-coded into this function
        self.ts = np.array([0., 2., 5., 15., 30., 60., 90.])

        # import function returns from model.py
        self.IL2Rb_species_IDX = getSurfaceIL2RbSpecies()

        # percentage value that is used in scaling output
        self.y_max = 10

    def singleCalc(self, unkVec, cytokine, conc):
        """ Calculates the surface IL2Rb over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc

        returnn, retVal = runCkineU(self.ts, unkVec)

        assert retVal >= 0

        a = np.dot(returnn, self.IL2Rb_species_IDX)

        return a / a[0]

    def calc(self, unkVec):
        '''This function uses an unkVec that has the same elements as the unkVec in fit.py'''

        assert unkVec.size == nParams()

        # set IL2 concentrations
        unkVecIL2RaMinus = unkVec.copy()
        unkVecIL2RaMinus[18] = 0.

        # calculate IL2 stimulation
        a = self.singleCalc(unkVec, 0, 1.)
        b = self.singleCalc(unkVec, 0, 500.)
        c = self.singleCalc(unkVecIL2RaMinus, 0, 1.)
        d = self.singleCalc(unkVecIL2RaMinus, 0, 500.)

        # calculate IL15 stimulation
        e = self.singleCalc(unkVec, 1, 1.)
        f = self.singleCalc(unkVec, 1, 500.)
        g = self.singleCalc(unkVecIL2RaMinus, 1, 1.)
        h = self.singleCalc(unkVecIL2RaMinus, 1, 500.)

        return np.concatenate((a, b, c, d, e, f, g, h))

    def plot_structure(self, IL2vec, IL15vec, title):
        plt.title(title)
        plt.scatter(self.ts, IL2vec, color='r', label='IL2', alpha=0.7)
        plt.scatter(self.ts, IL15vec, color='g', label='IL15', alpha=0.7)
        # plt.ylim(0,(y_max + (0.2 * y_max)))
        plt.ylabel("Surface IL2Rb (% x " + str(self.y_max) + ')')
        plt.xlabel("Time (min)")
        plt.legend()
        plt.show()

    def plot(self, unkVec):
        output = self.calc(unkVec) * self.y_max
        IL2_1_plus = output[0:7]
        IL2_500_plus = output[7:14]
        IL2_1_minus = output[14:21]
        IL2_500_minus = output[21:28]
        IL15_1_plus = output[28:35]
        IL15_500_plus = output[35:42]
        IL15_1_minus = output[42:49]
        IL15_500_minus = output[49:56]

        self.plot_structure(IL2_1_minus, IL15_1_minus, '1 nM and IL2Ra-')
        self.plot_structure(IL2_500_minus, IL15_500_minus, "500 nM and IL2Ra-")
        self.plot_structure(IL2_1_plus, IL15_1_plus, "1 nM and IL2Ra+")
        self.plot_structure(IL2_500_plus, IL15_500_plus, "500 nM and IL2Ra+")


class pstat:
    '''Generate values to match the pSTAT5 measurements used in fitting'''
    def __init__(self):
        self.PTS = 30
        self.cytokC = np.logspace(-3.3, 2.7, self.PTS) # 8 log-spaced values between our two endpoints
        
        # import function returns from model.py
        self.activity = getTotalActiveSpecies().astype(np.float64)
        
        self.ts = np.array([500.]) # was 500. in literature

        # percentage value that is used in scaling output
        self.y_max = 100

    def singleCalc(self, unkVec, cytokine, conc):
        """ Calculates the surface IL2Rb over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc

        returnn, retVal = runCkineU(self.ts, unkVec)

        assert retVal >= 0

        return np.dot(returnn, self.activity)

    def calc(self, unkVec):
        '''This function uses an unkVec that has the same elements as the unkVec in fit.py'''

        assert unkVec.size == nParams()

        unkVec_IL2Raminus = unkVec.copy()
        unkVec_IL2Raminus[18] = 0.0 # set IL2Ra expression rate to 0

        # Calculate activities
        actVec_IL2 = np.fromiter((self.singleCalc(unkVec, 0, x) for x in self.cytokC), np.float64)
        actVec_IL2_IL2Raminus = np.fromiter((self.singleCalc(unkVec_IL2Raminus, 0, x) for x in self.cytokC), np.float64)
        actVec_IL15 = np.fromiter((self.singleCalc(unkVec, 1, x) for x in self.cytokC), np.float64)
        actVec_IL15_IL2Raminus = np.fromiter((self.singleCalc(unkVec_IL2Raminus, 1, x) for x in self.cytokC), np.float64)

        # Normalize to the maximal activity, put together into one vector
        actVec = np.concatenate((actVec_IL2 / np.max(actVec_IL2), actVec_IL2_IL2Raminus / np.max(actVec_IL2_IL2Raminus), actVec_IL15 / np.max(actVec_IL15), actVec_IL15_IL2Raminus / np.max(actVec_IL15_IL2Raminus)))

        return actVec

    def plot_structure(self, IL2vec, IL15vec, title):
        plt.title(title)
        plt.scatter(np.log10(self.cytokC), IL2vec, color='r', alpha=0.5, label="IL2")
        plt.scatter(np.log10(self.cytokC), IL15vec, color='g', alpha=0.5, label='IL15')
        plt.ylim(0,(self.y_max + (0.25*self.y_max)))
        plt.ylabel('Maximal p-STAT5 (% x ' + str(self.y_max) + ')')
        plt.xlabel('log10 of cytokine concentration (nM)')
        plt.legend()
        plt.show()

    def plot(self, unkVec):
        output = self.calc(unkVec) * self.y_max
        IL2_plus = output[0:self.PTS]
        IL2_minus = output[self.PTS:(self.PTS*2)]
        IL15_plus = output[(self.PTS*2):(self.PTS*3)]
        IL15_minus = output[(self.PTS*3):(self.PTS*4)]

        self.plot_structure(IL2_minus, IL15_minus, "IL2Ra- YT-1 cells")
        self.plot_structure(IL2_plus, IL15_plus, "IL2Ra+ YT-1 cells")
