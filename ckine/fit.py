"""
This file includes the classes and functions necessary to fit the IL2 and IL15 model to the experimental data.
"""
from os.path import join, dirname, abspath
import pymc3 as pm
import theano.tensor as T
import numpy as np
import pandas as pds
from .model import getTotalActiveSpecies, getSurfaceIL2RbSpecies, receptor_expression
from .differencing_op import runCkineDoseOp


def load_data(filename):
    """ Return path of CSV files. """
    path = dirname(abspath(__file__))
    return pds.read_csv(join(path, filename)).values


def sampling(M):
    """ This is the sampling that actually runs the model. """
    return pm.sample(init="ADVI", chains=2, model=M)


def commonTraf():
    """ Set the common trafficking parameter priors. """
    kfwd = pm.Lognormal('kfwd', mu=np.log(0.001), sd=0.5, shape=1)
    endo = pm.Lognormal('endo', mu=np.log(0.1), sd=0.1, shape=1)
    activeEndo = pm.Lognormal('activeEndo', sd=0.1, shape=1)
    kRec = pm.Lognormal('kRec', mu=np.log(0.1), sd=0.1, shape=1)
    kDeg = pm.Lognormal('kDeg', mu=np.log(0.01), sd=0.2, shape=1)
    sortF = pm.Beta('sortF', alpha=12, beta=80, shape=1)
    return kfwd, endo, activeEndo, kRec, kDeg, sortF


def find_gc(traf=True, endo=0, kRec=0, sortF=0, kDeg=0):
    """ Calculates gc expression rate for YT-1 cells using data file and receptor_expression function. If 'traf' is false then we just return the receptor abundance level. """
    data = load_data("data/YT_1_receptor_levels.csv")  # data = [['YT-1', -1, -1, gc, -1]] where data[0, 3] represents the gc level
    if traf:
        return receptor_expression(data[0, 3], endo, kRec, sortF, kDeg)

    return data[0, 3]


class IL2Rb_trafficking:
    """ Calculating the percent of IL2Rb on cell surface under IL2 and IL15 stimulation according to Ring et al."""

    def __init__(self):
        numpy_data = load_data('data/IL2Ra+_surface_IL2RB_datasets.csv')
        numpy_data2 = load_data('data/IL2Ra-_surface_IL2RB_datasets.csv')

        # times from experiment are hard-coded into this function
        self.ts = np.array([0., 2., 5., 15., 30., 60., 90.])

        slicingg = (1, 5, 2, 6)

        # Concatted data
        self.data = np.concatenate((numpy_data[:, slicingg].flatten(order='F'), numpy_data2[:, slicingg].flatten(order='F'))) / 10.

        self.cytokM = np.zeros((4, 6), dtype=np.float64)
        self.cytokM[0, 0] = 1.
        self.cytokM[1, 0] = 500.
        self.cytokM[2, 1] = 1.
        self.cytokM[3, 1] = 500.

    def calc(self, unkVec):
        """ Calculates difference between relative IL2Rb on surface in model prediction and Ring experiment. """
        # Condense to just IL2Rb
        Op = runCkineDoseOp(tt=self.ts, condense=getSurfaceIL2RbSpecies().astype(np.float64), conditions=self.cytokM)

        # IL2Ra+ stimulation, IL2Ra- stimulation
        a = T.concatenate((Op(unkVec), Op(T.set_subtensor(unkVec[16], 0.0))))

        # return residual assuming all IL2Rb starts on the cell surface
        return a / a[0] - self.data


class IL2_15_activity:
    """ Calculating the pSTAT activity residuals for IL2 and IL15 stimulation in Ring et al. """

    def __init__(self):
        data = load_data('./data/IL2_IL15_extracted_data.csv')
        self.fit_data = np.concatenate((data[:, 6], data[:, 7], data[:, 2], data[:, 3])) / 100.  # the IL15_IL2Ra- data is within the 4th column (index 3)
        self.cytokC = np.logspace(-3.3, 2.7, 8)  # 8 log-spaced values between our two endpoints
        self.cytokM = np.zeros((self.cytokC.size * 2, 6), dtype=np.float64)
        self.cytokM[0:self.cytokC.size, 0] = self.cytokC
        self.cytokM[self.cytokC.size::, 1] = self.cytokC

    def calc(self, unkVec, scale):
        """ Simulate the STAT5 measurements and return residuals between model prediction and experimental data. """

        # IL2Ra- cells have same IL15 activity, so we can just reuse same solution
        Op = runCkineDoseOp(tt=np.array(500.), condense=getTotalActiveSpecies().astype(np.float64), conditions=self.cytokM)

        unkVecIL2RaMinus = T.set_subtensor(unkVec[16], 0.0)  # Set IL2Ra to zero

        # put together into one vector
        actCat = T.concatenate((Op(unkVec), Op(unkVecIL2RaMinus)))

        # account for pSTAT5 saturation
        actCat = actCat / (actCat + scale)

        # normalize from 0 to 1 and return the residual
        return self.fit_data - actCat / T.max(actCat)


class build_model:
    """ Build the overall model handling Ring et al. """

    def __init__(self, traf=True):
        self.traf = traf
        self.dst15 = IL2_15_activity()
        if self.traf:
            self.IL2Rb = IL2Rb_trafficking()
        self.M = self.build()

    def build(self):
        """The PyMC model that incorporates Bayesian Statistics in order to store what the likelihood of the model is for a given point."""
        M = pm.Model()

        with M:
            if self.traf:
                kfwd, endo, activeEndo, kRec, kDeg, sortF = commonTraf()
            else:
                kfwd = pm.Lognormal('kfwd', mu=np.log(0.001), sd=0.5, shape=1)
                # Assigning trafficking to zero to fit without trafficking
                endo = T.zeros(1, dtype=np.float64)
                activeEndo = T.zeros(1, dtype=np.float64)
                kRec = T.zeros(1, dtype=np.float64)
                kDeg = T.zeros(1, dtype=np.float64)
                sortF = T.ones(1, dtype=np.float64) * 0.5

            gc_value = find_gc(self.traf, endo, kRec, sortF, kDeg)  # find rate of gc expression or gc abundance (depending on traf)
            Rexpr_gc = T.ones(1, dtype=np.float64) * gc_value
            rxnrates = pm.Lognormal('rxn', sd=0.5, shape=6)  # 6 reverse rxn rates for IL2/IL15
            nullRates = T.ones(4, dtype=np.float64)  # k27rev, k31rev, k33rev, k35rev
            Rexpr_2Ra_2Rb = pm.Lognormal('Rexpr_2Ra_2Rb', sd=0.5, shape=2)  # Expression: IL2Ra, IL2Rb, gc
            Rexpr_15Ra = pm.Lognormal('Rexpr_15Ra', sd=0.5, shape=1)  # Expression: IL15Ra
            scale = pm.Lognormal('scales', mu=np.log(100.), sd=1, shape=1)  # create scaling constant for activity measurements

            unkVec = T.concatenate((kfwd, rxnrates, nullRates, endo, activeEndo, sortF, kRec, kDeg, Rexpr_2Ra_2Rb, Rexpr_gc, Rexpr_15Ra, nullRates * 0.0))

            Y_15 = self.dst15.calc(unkVec, scale)  # fitting the data based on dst15.calc for the given parameters
            sd_15 = T.minimum(T.std(Y_15), 0.03)  # Add bounds for the stderr to help force the fitting solution
            pm.Deterministic('Y_15', T.sum(T.square(Y_15)))
            pm.Normal('fitD_15', sd=sd_15, observed=Y_15)  # experimental-derived stderr is used

            if self.traf:
                Y_int = self.IL2Rb.calc(unkVec)  # fitting the data based on dst.calc for the given parameters
                sd_int = T.minimum(T.std(Y_int), 0.02)  # Add bounds for the stderr to help force the fitting solution
                pm.Deterministic('Y_int', T.sum(T.square(Y_int)))
                pm.Normal('fitD_int', sd=sd_int, observed=Y_int)

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M
