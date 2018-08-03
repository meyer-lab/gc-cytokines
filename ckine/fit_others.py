"""
This file includes the classes and functions necessary to fit the IL4 and IL7 model to experimental data.
"""
from os.path import join
import pymc3 as pm, theano.tensor as T, os
import numpy as np, pandas as pds
from .model import getTotalActiveSpecies
from .differencing_op import runCkineDoseOp

class IL4_7_activity:
    """ This class is responsible for calculating residuals between model predictions and the data from Gonnord et al. """
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        dataIL4 = pds.read_csv(join(path, "./data/Gonnord_S3B.csv")).values # imports IL4 file into pandas array
        dataIL7 = pds.read_csv(join(path, "./data/Gonnord_S3C.csv")).values # imports IL7 file into pandas array

        # units are converted from pg/mL to nM
        self.cytokC_4 = np.array([5., 50., 500., 5000., 50000., 250000.]) / 14900. # 14.9 kDa according to sigma aldrich
        self.cytokC_7 = np.array([1., 10., 100., 1000., 10000., 100000.]) / 17400. # 17.4 kDa according to prospec bio

        self.cytokM = np.zeros((self.cytokC_4.size*2, 6), dtype=np.float64)
        self.cytokM[0:self.cytokC_4.size, 4] = self.cytokC_4
        self.cytokM[self.cytokC_4.size::, 2] = self.cytokC_7

        self.fit_data = np.concatenate((dataIL4[:, 1], dataIL4[:, 2], dataIL7[:, 1], dataIL7[:, 2])) # measurements aren't normalized
        self.activity = getTotalActiveSpecies().astype(np.float64)


    def calc(self, unkVec, scales):
        """ Simulate the experiment with different ligand stimulations and compare with experimental data. """
        Op = runCkineDoseOp(tt=np.array(10.), condense=getTotalActiveSpecies().astype(np.float64), conditions=self.cytokM)

        # Run the experiment
        outt = Op(unkVec)

        actVecIL4 = outt[0:self.cytokC_4.size]
        actVecIL7 = outt[self.cytokC_4.size:self.cytokC_4.size*2]

        # Multiply by scaling constants and put together in one vector
        actVec = T.concatenate((actVecIL4 * scales[0], actVecIL4 * scales[0], actVecIL7 * scales[1], actVecIL7 * scales[1]))

        # return residual
        return self.fit_data - actVec


class build_model:
    """Going to load the data from the CSV file at the very beginning of when build_model is called... needs to be separate member function to avoid uploading file thousands of times."""
    def __init__(self):
        self.act = IL4_7_activity()
        self.M = self.build()

    def build(self):
        """The PyMC model that incorporates Bayesian Statistics in order to store what the likelihood of the model is for a given point."""
        M = pm.Model()

        with M:
            kfwd = pm.Lognormal('kfwd', mu=np.log(0.00001), sd=1, shape=1)
            nullRates = T.ones(6, dtype=np.float64) # associated with IL2 and IL15
            Tone = T.ones(1, dtype=np.float64)
            Tzero = T.zeros(1, dtype=np.float64)
            k27rev = pm.Lognormal('k27rev', mu=np.log(0.1), sd=10, shape=1) # associated with IL7
            k33rev = pm.Lognormal('k33rev', mu=np.log(0.1), sd=10, shape=1) # associated with IL4
            endo_activeEndo = pm.Lognormal('endo', mu=np.log(0.1), sd=0.1, shape=2)
            sortF = pm.Beta('sortF', alpha=20, beta=40, testval=0.333, shape=1)*0.95
            kRec_kDeg = pm.Lognormal('kRec_kDeg', mu=np.log(0.1), sd=0.1, shape=2)
            GCexpr = (328. * endo_activeEndo[0]) / (1. + ((kRec_kDeg[0]*(1.-sortF)) / (kRec_kDeg[1]*sortF))) # constant according to measured number per cell
            IL7Raexpr = (2591. * endo_activeEndo[0]) / (1. + ((kRec_kDeg[0]*(1.-sortF)) / (kRec_kDeg[1]*sortF))) # constant according to measured number per cell
            IL4Raexpr = (254. * endo_activeEndo[0]) / (1. + ((kRec_kDeg[0]*(1.-sortF)) / (kRec_kDeg[1]*sortF))) # constant according to measured number per cell
            scales = pm.Lognormal('scales', mu=np.log(100), sd=np.log(25), shape=2) # create scaling constants for activity measurements

            unkVec = T.concatenate((kfwd, nullRates, k27rev, Tone, k33rev, Tone, endo_activeEndo, sortF, kRec_kDeg))
            unkVec = T.concatenate((unkVec, Tzero, Tzero, GCexpr, Tzero, IL7Raexpr, Tzero, IL4Raexpr, Tzero)) # indexing same as in model.hpp

            Y_int = self.act.calc(unkVec, scales) # fitting the data based on act.calc for the given parameters

            pm.Deterministic('Y_int', T.sum(T.square(Y_int)))

            pm.Normal('fitD_int', sd=700, observed=Y_int)

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M

    def sampling(self):
        """This is the sampling that actually runs the model."""
        self.trace = pm.sample(init='advi', model=self.M)

    def fit_ADVI(self):
        """ Running fit_advi instead of true sampling. """
        with self.M:
            approx = pm.fit(40000, method='fullrank_advi')
            self.trace = approx.sample()

    def profile(self):
        """ Profile the gradient calculation. """
        self.M.profile(pm.theanof.gradient(self.M.logpt, None)).summary()

