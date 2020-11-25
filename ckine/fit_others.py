"""
This file includes the classes and functions necessary to fit the IL4 and IL7 model to experimental data.
"""
import os
from os.path import join
import pymc3 as pm
import theano.tensor as T
import numpy as np
import pandas as pds
from .model import getTotalActiveSpecies, nSpecies, getActiveSpecies, internalStrength, halfL, getCytokineSpecies
from .differencing_op import runCkineDoseOp
from .fit import commonTraf


class IL4_7_activity:  # pylint: disable=too-few-public-methods
    """ This class is responsible for calculating residuals between model predictions and the data from Gonnord figure S3B/C """

    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.dataIL4 = pds.read_csv(join(path, "./data/Gonnord_S3B.csv")).values[:, 1:]  # imports IL4 file into pandas array
        self.dataIL4 /= np.amax(self.dataIL4)
        self.dataIL7 = pds.read_csv(join(path, "./data/Gonnord_S3C.csv")).values[:, 1:]  # imports IL7 file into pandas array
        self.dataIL7 /= np.amax(self.dataIL7)
        self.nDoses = 6

        self.cytokM = np.zeros((self.nDoses * 2, 6), dtype=np.float64)
        # units are converted from pg/mL to nM
        self.cytokM[0: self.nDoses, 4] = np.array([5.0, 50.0, 500.0, 5000.0, 50000.0, 250000.0]) / 14900.0  # 14.9 kDa according to sigma aldrich
        self.cytokM[self.nDoses::, 2] = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]) / 17400.0  # 17.4 kDa according to prospec bio

    def calc(self, unkVec, M):
        """ Simulate the experiment with different ligand stimulations and compare with experimental data. """
        Op = runCkineDoseOp(tt=np.array(10.0), condense=getTotalActiveSpecies().astype(np.float64), conditions=self.cytokM)

        # Run the experiment
        outt = Op(unkVec)

        actVecIL4 = outt[0: self.nDoses]
        actVecIL7 = outt[self.nDoses: self.nDoses * 2]

        # normalize each actVec by its maximum
        actVecIL4 = actVecIL4 / T.max(actVecIL4)
        actVecIL7 = actVecIL7 / T.max(actVecIL7)
        actVecIL4 = T.tile(actVecIL4, (2, 1))
        actVecIL7 = T.tile(actVecIL7, (2, 1))

        # put into one vector
        actVecIL4 = T.flatten(self.dataIL4.T - actVecIL4)
        actVecIL7 = T.flatten(self.dataIL7.T - actVecIL7)
        Y_int = T.concatenate((actVecIL4, actVecIL7))

        with M:
            pm.Deterministic("Y_int", T.sum(T.square(Y_int)))
            pm.Normal("fitD_int", sigma=T.minimum(T.std(Y_int), 0.1), observed=Y_int)


class crosstalk:  # pylint: disable=too-few-public-methods
    """ This class performs the calculations necessary in order to fit our model to Gonnord Fig S3D. """

    def __init__(self):
        self.ts = np.array([10.0])  # was 10. in literature
        self.cytokM = np.zeros((2, 6), dtype=np.float64)
        self.cytokM[0, 4] = 100.0 / 14900.0  # concentration used for IL4 stimulation
        self.cytokM[1, 2] = 50.0 / 17400.0  # concentration used for IL7 stimulation

        path = os.path.dirname(os.path.abspath(__file__))
        self.data = pds.read_csv(join(path, "./data/Gonnord_S3D.csv")).values
        data = self.data
        self.fit_data = np.concatenate((data[:, 1], data[:, 2], data[:, 3], data[:, 6], data[:, 7], data[:, 8])) / 100.0
        self.pre_IL7 = data[:, 0] / 17400.0  # concentrations of IL7 used as pretreatment
        self.pre_IL4 = data[:, 5] / 14900.0  # concentrations of IL4 used as pretreatment

    def singleCalc(self, unkVec, pre_cytokine, pre_conc, stim_cytokine, stim_conc):
        """ This function generates the active vector for a given unkVec, cytokine used for inhibition and concentration of pretreatment cytokine. """
        unkVec2 = T.set_subtensor(unkVec[stim_cytokine], stim_conc)

        # create ligands array for stimulation at t=0
        ligands = np.zeros((1, 6))
        ligands[0, stim_cytokine] = stim_conc
        ligands[0, pre_cytokine] = pre_conc  # assume pretreatment ligand stays constant

        prelig = np.zeros(6)
        prelig[pre_cytokine] = pre_conc

        condvec = np.zeros(nSpecies())
        condvec[np.where(getActiveSpecies())] = 1.0
        condvec[np.array(np.where(getActiveSpecies())) + halfL()] = internalStrength()
        condvecCyto = np.zeros(nSpecies())
        condvecCyto[getCytokineSpecies()[stim_cytokine]] = condvec[getCytokineSpecies()[stim_cytokine]]
        condvecCyto[getCytokineSpecies()[stim_cytokine] + halfL()] = condvec[getCytokineSpecies()[stim_cytokine] + halfL()]

        Op = runCkineDoseOp(preT=self.ts, tt=self.ts, conditions=ligands, condense=condvecCyto, prestim=prelig)

        # perform the experiment
        outt = Op(unkVec2)

        return outt[0]  # only look at active species associated with stimulation cytokine

    def singleCalc_no_pre(self, unkVec):
        """ This function generates the active vector for a given unkVec, cytokine, and concentration. """
        Op = runCkineDoseOp(tt=self.ts, condense=getTotalActiveSpecies().astype(np.float64), conditions=self.cytokM)

        # Run the experiment
        outt = Op(unkVec)

        actVecIL4 = outt[0]
        actVecIL7 = outt[1]
        return actVecIL4, actVecIL7

    def calc(self, unkVec):
        """ Generates residual calculation that compares model to data. """
        # with no pretreatment
        IL4stim_no_pre, IL7stim_no_pre = self.singleCalc_no_pre(unkVec)

        # IL7 pretreatment with IL4 stimulation
        actVec_IL4stim = T.stack((list(self.singleCalc(unkVec, 2, x, 4, self.cytokM[0, 4]) for x in self.pre_IL7)))

        # IL4 pretreatment with IL7 stimulation
        actVec_IL7stim = T.stack((list(self.singleCalc(unkVec, 4, x, 2, self.cytokM[1, 2]) for x in self.pre_IL4)))

        case1 = 1 - (actVec_IL4stim / IL4stim_no_pre)  # % inhibition of IL4 act. after IL7 pre.
        case2 = 1 - (actVec_IL7stim / IL7stim_no_pre)  # % inhibition of IL7 act. after IL4 pre.
        inh_vec = T.concatenate((case1, case1, case1, case2, case2, case2))  # mimic order of CSV file

        return inh_vec - self.fit_data


class build_model:
    """ Build a model that minimizes residuals in above classes by using MCMC to find optimal rate parameters. """

    def __init__(self, pretreat=True):
        self.act = IL4_7_activity()
        self.cross = crosstalk()
        self.pretreat = pretreat
        self.M = self.build()

    def build(self):
        """The PyMC model that incorporates Bayesian Statistics in order to store what the likelihood of the model is for a given point."""
        M = pm.Model()

        with M:
            kfwd, endo, activeEndo, kRec, kDeg, sortF = commonTraf()
            nullRates = T.ones(6, dtype=np.float64)  # associated with IL2 and IL15
            Tone = T.ones(1, dtype=np.float64)
            k27rev = pm.Lognormal("k27rev", mu=np.log(0.1), sigma=1, shape=1)  # associated with IL7
            k33rev = pm.Lognormal("k33rev", mu=np.log(0.1), sigma=1, shape=1)  # associated with IL4

            # constant according to measured number per cell. gc, blank, IL7R, blank, IL4R
            Rexpr = (np.array([0.0, 0.0, 328.0, 0.0, 2591.0, 0.0, 254.0, 0.0]) * endo) / (1.0 + ((kRec * (1.0 - sortF)) / (kDeg * sortF)))

            # indexing same as in model.hpp
            unkVec = T.concatenate((kfwd, nullRates, k27rev, Tone, k33rev, Tone, endo, activeEndo, sortF, kRec, kDeg, Rexpr))

            self.act.calc(unkVec, M)  # fitting the data based on act.calc for the given parameters

            if self.pretreat is True:
                Y_cross = self.cross.calc(unkVec)  # fitting the data based on cross.calc
                pm.Deterministic("Y_cross", T.sum(T.square(Y_cross)))
                sd_cross = T.minimum(T.std(Y_cross), 0.1)
                pm.Normal("fitD_cross", sigma=sd_cross, observed=Y_cross)  # the stderr is definitely less than 0.2

            # Save likelihood
            pm.Deterministic("logp", M.logpt)

        return M
