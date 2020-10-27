"""
This file includes the classes and functions necessary to fit the IL2 and IL15 model to the experimental data.
"""
from os.path import join, dirname, abspath
import pymc3 as pm
import theano.tensor as T
import numpy as np
import pandas as pds
from .model import getTotalActiveSpecies, getSurfaceIL2RbSpecies, getSurfaceGCSpecies
from .differencing_op import runCkineDoseOp


def load_data(filename):
    """ Return path of CSV files. """
    path = dirname(abspath(__file__))
    return pds.read_csv(join(path, filename)).values


def sampling(M):
    """ This is the sampling that actually runs the model. """
    return pm.sample(init="adapt_diag", chains=2, model=M, target_accept=0.90, step_scale=0.01)


def commonTraf(trafficking=True):
    """ Set the common trafficking parameter priors. """
    kfwd = pm.Lognormal("kfwd", mu=np.log(0.001), sigma=0.5, shape=1)

    if trafficking:
        endo = pm.Lognormal("endo", mu=np.log(0.1), sigma=0.1, shape=1)
        activeEndo = pm.Lognormal("activeEndo", sigma=0.1, shape=1)
        kRec = pm.Lognormal("kRec", mu=np.log(0.1), sigma=0.1, shape=1)
        kDeg = pm.Lognormal("kDeg", mu=np.log(0.01), sigma=0.2, shape=1)
        sortF = pm.Beta("sortF", alpha=12, beta=80, shape=1)
    else:
        # Assigning trafficking to zero to fit without trafficking
        endo = activeEndo = kRec = kDeg = T.zeros(1, dtype=np.float64)
        sortF = T.ones(1, dtype=np.float64) * 0.5
    return kfwd, endo, activeEndo, kRec, kDeg, sortF


class IL2Rb_trafficking:  # pylint: disable=too-few-public-methods
    """ Calculating the percent of IL2Rb on cell surface under IL2 and IL15 stimulation according to Ring et al."""

    def __init__(self):
        numpy_data = load_data("data/IL2Ra+_surface_IL2RB_datasets.csv")
        numpy_data2 = load_data("data/IL2Ra-_surface_IL2RB_datasets.csv")

        # times from experiment are hard-coded into this function
        self.ts = np.array([0.0, 2.0, 5.0, 15.0, 30.0, 60.0, 90.0])

        slicingg = (1, 5, 2, 6)

        # Concatted data
        self.data = np.concatenate((numpy_data[:, slicingg].flatten(order="F"), numpy_data2[:, slicingg].flatten(order="F"))) / 10.0

        self.cytokM = np.zeros((4, 6), dtype=np.float64)
        self.cytokM[0, 0] = 1.0
        self.cytokM[1, 0] = 500.0
        self.cytokM[2, 1] = 1.0
        self.cytokM[3, 1] = 500.0

    def calc(self, unkVec):
        """ Calculates difference between relative IL2Rb on surface in model prediction and Ring experiment. """
        # Condense to just IL2Rb
        Op = runCkineDoseOp(tt=self.ts, condense=getSurfaceIL2RbSpecies().astype(np.float64), conditions=self.cytokM)

        # IL2Ra+ stimulation, IL2Ra- stimulation
        a = T.concatenate((Op(unkVec), Op(T.set_subtensor(unkVec[16], 0.0))))

        # return residual assuming all IL2Rb starts on the cell surface
        return a / a[0] - self.data


class gc_trafficking:  # pylint: disable=too-few-public-methods
    """ Calculating the percent of gc on cell surface under 157 nM of IL2 stimulation according to Mitra et al."""

    def __init__(self):
        numpy_data = load_data("data/mitra_surface_gc_depletion.csv")

        # times from experiment are in first column
        self.ts = numpy_data[:, 0]

        # percent of gc that stays on surface (scale from 0-1)
        self.data = numpy_data[:, 1] / 100.0

        self.cytokM = np.zeros((1, 6), dtype=np.float64)
        self.cytokM[0, 0] = 1000.0  # 1 uM of IL-2 was given to 3 x 10^5 YT-1 cells

    def calc(self, unkVec):
        """ Calculates difference between relative IL2Rb on surface in model prediction and Ring experiment. """
        # Condense to just gc
        Op = runCkineDoseOp(tt=self.ts, condense=getSurfaceGCSpecies().astype(np.float64), conditions=self.cytokM)

        # IL2Ra+ stimulation only
        a = Op(unkVec)

        # return residual assuming all gc starts on the cell surface
        return a / a[0] - self.data


class IL2_15_activity:  # pylint: disable=too-few-public-methods
    """ Calculating the pSTAT activity residuals for IL2 and IL15 stimulation in Ring et al. """

    def __init__(self):
        data = load_data("./data/IL2_IL15_extracted_data.csv")
        self.fit_data = np.concatenate((data[:, 6], data[:, 7], data[:, 2], data[:, 3])) / 100.0  # the IL15_IL2Ra- data is within the 4th column (index 3)
        self.cytokC = np.logspace(-3.3, 2.7, 8)  # 8 log-spaced values between our two endpoints
        self.cytokM = np.zeros((self.cytokC.size * 2, 6), dtype=np.float64)
        self.cytokM[0: self.cytokC.size, 0] = self.cytokC
        self.cytokM[self.cytokC.size::, 1] = self.cytokC

    def calc(self, unkVec, scale):
        """ Simulate the STAT5 measurements and return residuals between model prediction and experimental data. """

        # IL2Ra- cells have same IL15 activity, so we can just reuse same solution
        Op = runCkineDoseOp(tt=np.array(500.0), condense=getTotalActiveSpecies().astype(np.float64), conditions=self.cytokM)

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
            self.gc = gc_trafficking()
        self.M = self.build()

    def build(self):
        """The PyMC model that incorporates Bayesian Statistics in order to store what the likelihood of the model is for a given point."""
        M = pm.Model()

        with M:
            kfwd, endo, activeEndo, kRec, kDeg, sortF = commonTraf(trafficking=self.traf)

            rxnrates = pm.Lognormal("rxn", sigma=0.5, shape=6)  # 6 reverse rxn rates for IL2/IL15
            nullRates = T.ones(4, dtype=np.float64)  # k27rev, k31rev, k33rev, k35rev
            Rexpr_2Ra = pm.Lognormal("Rexpr_2Ra", sigma=0.5, shape=1)  # Expression: IL2Ra
            Rexpr_2Rb = pm.Lognormal("Rexpr_2Rb", sigma=0.5, shape=1)  # Expression: IL2Rb
            Rexpr_15Ra = pm.Lognormal("Rexpr_15Ra", sigma=0.5, shape=1)  # Expression: IL15Ra
            Rexpr_gc = pm.Lognormal("Rexpr_gc", sigma=0.5, shape=1)  # Expression: gamma chain
            scale = pm.Lognormal("scales", mu=np.log(100.0), sigma=1, shape=1)  # create scaling constant for activity measurements

            unkVec = T.concatenate((kfwd, rxnrates, nullRates, endo, activeEndo, sortF, kRec, kDeg, Rexpr_2Ra, Rexpr_2Rb, Rexpr_gc, Rexpr_15Ra, nullRates * 0.0))
            unkVec_2Ra_minus = T.concatenate((kfwd, rxnrates, nullRates, endo, activeEndo, sortF, kRec, kDeg, T.zeros(1, dtype=np.float64), Rexpr_2Rb, Rexpr_gc, Rexpr_15Ra, nullRates * 0.0))

            Y_15 = self.dst15.calc(unkVec, scale)  # fitting the data based on dst15.calc for the given parameters
            sd_15 = T.minimum(T.std(Y_15), 0.03)  # Add bounds for the stderr to help force the fitting solution
            pm.Deterministic("Y_15", T.sum(T.square(Y_15)))
            pm.Normal("fitD_15", sigma=sd_15, observed=Y_15)  # experimental-derived stderr is used

            if self.traf:
                Y_int = self.IL2Rb.calc(unkVec)  # fitting the data based on IL2Rb surface data
                sd_int = T.minimum(T.std(Y_int), 0.02)  # Add bounds for the stderr to help force the fitting solution
                pm.Deterministic("Y_int", T.sum(T.square(Y_int)))
                pm.Normal("fitD_int", sigma=sd_int, observed=Y_int)

                Y_gc = self.gc.calc(unkVec_2Ra_minus)  # fitting the data using IL2Ra- cells
                sd_gc = T.minimum(T.std(Y_gc), 0.02)  # Add bounds for the stderr to help force the fitting solution
                pm.Deterministic("Y_gc", T.sum(T.square(Y_gc)))
                pm.Normal("fitD_gc", sigma=sd_gc, observed=Y_gc)

            # Save likelihood
            pm.Deterministic("logp", M.logpt)

        return M
