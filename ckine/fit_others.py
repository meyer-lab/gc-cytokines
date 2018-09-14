"""
This file includes the classes and functions necessary to fit the IL4 and IL7 model to experimental data.
"""
from os.path import join
import theano
import pymc3 as pm, theano.tensor as T, os
import numpy as np, pandas as pds
from .model import getTotalActiveSpecies, nParams, getTotalActiveCytokine
from .differencing_op import runCkineDoseOp, runCkinePreSOp

class IL4_7_activity:
    """ This class is responsible for calculating residuals between model predictions and the data from Gonnord figure S3B/C """
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

        IL4_data_max = np.amax(np.concatenate((dataIL4[:,1], dataIL4[:,2])))
        IL7_data_max = np.amax(np.concatenate((dataIL7[:,1], dataIL7[:,2])))
        self.fit_data = np.concatenate((dataIL4[:, 1] / IL4_data_max, dataIL4[:, 2] / IL4_data_max, dataIL7[:, 1] / IL7_data_max, dataIL7[:, 2] / IL7_data_max)) # measurements are normalized to max of own species

    def calc(self, unkVec, scales):
        """ Simulate the experiment with different ligand stimulations and compare with experimental data. """
        Op = runCkineDoseOp(tt=np.array(10.), condense=getTotalActiveSpecies().astype(np.float64), conditions=self.cytokM)

        # Run the experiment
        outt = Op(unkVec)

        actVecIL4 = outt[0:self.cytokC_4.size]
        actVecIL7 = outt[self.cytokC_4.size:self.cytokC_4.size*2]

        # normalize each actVec by its maximum
        actVecIL4 = actVecIL4 / T.max(actVecIL4)
        actVecIL7 = actVecIL7 / T.max(actVecIL7)

        # incorporate IC50 scale
        actVecIL4 = actVecIL4  / (actVecIL4 + scales[0])
        actVecIL7 = actVecIL7 / (actVecIL7 + scales[1])

        # concatenate into one vector
        actVec = T.concatenate((actVecIL4, actVecIL4, actVecIL7, actVecIL7))

        # return residual
        return self.fit_data - actVec

class crosstalk:
    """ This class performs the calculations necessary in order to fit our model to Gonnord Fig S3D. """
    def __init__(self):
        self.ts = np.array([10.]) # was 10. in literature
        self.cytokM = np.zeros((2, 6), dtype=np.float64)
        self.cytokM[0, 4] = 100. / 14900. # concentration used for IL4 stimulation
        self.cytokM[1, 2] = 50. / 17400. # concentration used for IL7 stimulation

        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(join(path, "./data/Gonnord_S3D.csv")).values
        self.fit_data = np.concatenate((data[:, 1], data[:, 2], data[:, 3], data[:, 6], data[:, 7], data[:, 8]))
        self.pre_IL7 = data[:, 0]   # concentrations of IL7 used as pretreatment
        self.pre_IL4 = data[:, 5]   # concentrations of IL4 used as pretreatment


    def singleCalc(self, unkVec, pre_cytokine, pre_conc, stim_cytokine, stim_conc):
        """ This function generates the active vector for a given unkVec, cytokine used for inhibition and concentration of pretreatment cytokine. """
        unkVec2 = T.set_subtensor(unkVec[pre_cytokine], pre_conc)

        # create ligands array for stimulation at t=0
        ligands = np.zeros((6))
        ligands[stim_cytokine] = stim_conc
        ligands[pre_cytokine] = pre_conc    # assume pretreatment ligand stays constant

        Op = runCkinePreSOp(tpre=self.ts, ts=self.ts, postlig=ligands)

        # perform the experiment
        outt = Op(unkVec2)

        return getTotalActiveCytokine(stim_cytokine, outt) # only look at active species associated with stimulation cytokine

    def singleCalc_no_pre(self, unkVec):
        ''' This function generates the active vector for a given unkVec, cytokine, and concentration. '''
        Op = runCkineDoseOp(tt=self.ts, condense=getTotalActiveSpecies().astype(np.float64), conditions=self.cytokM)

        # Run the experiment
        outt = Op(unkVec)

        actVecIL4 = outt[0]
        actVecIL7 = outt[1]
        return actVecIL4, actVecIL7


    def calc(self, unkVec, scales):
        """ Generates residual calculation that compares model to data. """
        # with no pretreatment
        IL4stim_no_pre, IL7stim_no_pre = self.singleCalc_no_pre(unkVec)

        # incorporate IC50
        IL4stim_no_pre = IL4stim_no_pre  / (IL4stim_no_pre + scales[0])
        IL7stim_no_pre = IL7stim_no_pre  / (IL7stim_no_pre + scales[1])

        # add 6 zeros to front of unkVec to handle ligands within unkVec
        unkVec = T.concatenate((T.zeros(6, dtype=np.float64), unkVec))

        # IL7 pretreatment with IL4 stimulation
        actVec_IL4stim = T.stack((list(self.singleCalc(unkVec, 2, x, 4, self.cytokM[0, 4]) for x in self.pre_IL7)))

        # IL4 pretreatment with IL7 stimulation
        actVec_IL7stim = T.stack((list(self.singleCalc(unkVec, 4, x, 2, self.cytokM[1, 2]) for x in self.pre_IL4)))

        # incorporate IC50
        actVec_IL4stim = actVec_IL4stim  / (actVec_IL4stim + scales[0])
        actVec_IL7stim = actVec_IL7stim  / (actVec_IL7stim + scales[1])


        case1 = (1-(actVec_IL4stim/IL4stim_no_pre)) * 100.    # % inhibition of IL4 act. after IL7 pre.
        case2 = (1-(actVec_IL7stim/IL7stim_no_pre)) * 100.    # % inhibition of IL7 act. after IL4 pre.
        inh_vec = T.concatenate((case1, case1, case1, case2, case2, case2 ))   # mimic order of CSV file

        return inh_vec - self.fit_data


class build_model:
    """ Build a model that minimizes residuals in above classes by using MCMC to find optimal rate parameters. """
    def __init__(self, pretreat=False):
        self.act = IL4_7_activity()
        self.cross = crosstalk()
        self.pretreat = pretreat
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
            scales = pm.Lognormal('scales', mu=np.log(1000.), sd=10, shape=2) # create scaling constants for activity measurements

            unkVec = T.concatenate((kfwd, nullRates, k27rev, Tone, k33rev, Tone, endo_activeEndo, sortF, kRec_kDeg))
            unkVec = T.concatenate((unkVec, Tzero, Tzero, GCexpr, Tzero, IL7Raexpr, Tzero, IL4Raexpr, Tzero)) # indexing same as in model.hpp

            Y_int = self.act.calc(unkVec, scales) # fitting the data based on act.calc for the given parameters

            pm.Deterministic('Y_int', T.sum(T.square(Y_int)))

            pm.Normal('fitD_int', sd=0.1, observed=Y_int)

            if self.pretreat == True:
                Y_cross = self.cross.calc(unkVec, scales)   # fitting the data based on cross.calc
                pm.Deterministic('Y_cross', T.sum(T.square(Y_cross)))
                pm.Normal('fitD_cross', sd=5, observed=Y_cross)

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M

    def sampling(self):
        """This is the sampling that actually runs the model."""
        self.trace = pm.sample(init='advi', model=self.M, cores=1, chains=1)

    def fit_ADVI(self):
        """ Running fit_advi instead of true sampling. """
        with self.M:
            approx = pm.fit(40000, method='fullrank_advi')
            self.trace = approx.sample()

    def profile(self):
        """ Profile the gradient calculation. """
        self.M.profile(pm.theanof.gradient(self.M.logpt, None)).summary()
