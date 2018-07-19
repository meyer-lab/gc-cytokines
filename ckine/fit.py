"""
This file includes the classes and functions necessary to fit the IL2 and IL15 model to the experimental data.
"""
from os.path import join, dirname, abspath
import pymc3 as pm
import theano
import theano.tensor as T
import numpy as np, pandas as pds
from .model import getTotalActiveSpecies, getSurfaceIL2RbSpecies
from .differencing_op import runCkineOp, runCkineKineticOp, runCkineDoseOp


def load_data(filename): 
    path = dirname(abspath(__file__))
    return pds.read_csv(join(path, filename)).values


class IL2Rb_trafficking:
    def __init__(self):
        # all of the IL2Rb trafficking data with IL2Ra+... first row contains headers... 9 columns and 8 rows... first column is time
        numpy_data = load_data('data/IL2Ra+_surface_IL2RB_datasets.csv')
        # all of the IL2Rb trafficking data with IL2Ra-... first row contains headers... 9 columns and 8 rows... first column is time
        numpy_data2 = load_data('data/IL2Ra-_surface_IL2RB_datasets.csv')

        # times from experiment are hard-coded into this function
        self.ts = np.array([0., 2., 5., 15., 30., 60., 90.])
        
        slicingg = np.array([1, 5, 2, 6])

        # Concatted data
        self.data = np.concatenate((numpy_data[:, slicingg], numpy_data2[:, slicingg])).flatten(order='F')/10.

    def calc(self, unkVec):
        unkVecIL2RaMinus = T.set_subtensor(unkVec[22], 0.0) # Set IL2Ra to zero

        # Condense to just IL2Rb
        KineticOp = runCkineKineticOp(self.ts, getSurfaceIL2RbSpecies())

        # IL2 stimulation
        a = KineticOp(T.set_subtensor(unkVec[0], 1.)) # col 2 of numpy_data has all the 1nM IL2Ra+ data
        b = KineticOp(T.set_subtensor(unkVec[0], 500.)) # col 6 of numpy_data has all the 500 nM IL2Ra+ data
        c = KineticOp(T.set_subtensor(unkVecIL2RaMinus[0], 1.)) # col 2 of numpy_data2 has all the 1nM IL2Ra- data
        d = KineticOp(T.set_subtensor(unkVecIL2RaMinus[0], 500.)) # col 6 of numpy_data2 has all the 500 nM IL2Ra- data
        # IL15 stimulation
        e = KineticOp(T.set_subtensor(unkVec[1], 1.))
        f = KineticOp(T.set_subtensor(unkVec[1], 500.))
        g = KineticOp(T.set_subtensor(unkVecIL2RaMinus[1], 1.))
        h = KineticOp(T.set_subtensor(unkVecIL2RaMinus[1], 500.))

        # assuming all IL2Rb starts on the cell surface
        return T.concatenate((a, b, e, f, c, d, g, h)) / a[0] - self.data

# this takes all the desired IL2 values we want to test and gives us the maximum activity value
# IL2 values pretty much ranged from 5 x 10**-4 to 500 nm with 8 points in between
# need the theano decorator to get around the fact that there are if-else statements when running odeint but
# we don't necessarily know the values for the rxn rates when we call our model
class IL2_15_activity:
    def __init__(self):
        """ This loads the experiment data and saves it as a member matrix and it also makes a vector of the IL15 concentrations that we are going to take care of. """
        path = dirname(abspath(__file__))
        data = load_data('./data/IL2_IL15_extracted_data.csv')
        self.cytokC = np.logspace(-3.3, 2.7, 8) # 8 log-spaced values between our two endpoints

        self.cytokM = np.zeros((self.cytokC.size*2, 6), dtype=np.float64)
        self.cytokM[0:self.cytokC.size, 0] = self.cytokC
        self.cytokM[self.cytokC.size::, 1] = self.cytokC

        self.fit_data = np.concatenate((data[:, 6], data[:, 7], data[:, 2], data[:, 3])) / 100. #the IL15_IL2Ra- data is within the 4th column (index 3)

    def calc(self, unkVec):
        """ Simulate the STAT5 measurements. """
        # We don't need the ligands in unkVec for this Op
        unkVec = unkVec[6::]

        # IL2Ra- cells have same IL15 activity, so we can just reuse same solution
        Op = runCkineDoseOp(tt=np.array(500.), condense=getTotalActiveSpecies().astype(np.float64), conditions=self.cytokM)

        unkVecIL2RaMinus = T.set_subtensor(unkVec[22], 0.0) # Set IL2Ra to zero

        # Normalize to the maximal activity, put together into one vector
        actCat = T.concatenate((Op(unkVec), Op(unkVecIL2RaMinus)))

        # value we're trying to minimize is the distance between the y-values on points of the graph that correspond to the same IL2 values
        return self.fit_data - (actCat / T.max(actCat))


class build_model:
    """ Build the overall model handling Ring et al. """
    # Going to load the data from the CSV file at the very beginning of when build_model is called...
    # needs to be separate member function to avoid uploading file thousands of times.
    def __init__(self):
        self.dst15 = IL2_15_activity()
        self.IL2Rb = IL2Rb_trafficking()
        self.M = self.build()

    def build(self):
        """The PyMC model that incorporates Bayesian Statistics in order to store what the likelihood of the model is for a given point."""
        M = pm.Model()

        with M:
            kfwd = pm.Lognormal('kfwd', mu=np.log(0.00001), sd=1, shape=1)
            rxnrates = pm.Lognormal('rxn', mu=np.log(0.1), sd=1, shape=6) # there are 6 reverse rxn rates associated with IL2 and IL15
            nullRates = T.ones(4, dtype=np.float64) # k27rev, k31rev, k33rev, k35rev
            endo_activeEndo = pm.Lognormal('endo', mu=np.log(0.1), sd=0.1, shape=2)
            kRec_kDeg = pm.Lognormal('kRec_kDeg', mu=np.log(0.1), sd=0.1, shape=2)
            Rexpr = pm.Lognormal('IL2Raexpr', sd=0.1, shape=4) # Expression: IL2Ra, IL2Rb, gc, IL15Ra
            sortF = pm.Beta('sortF', alpha=20, beta=40, testval=0.333, shape=1)*0.95

            ligands = T.zeros(6, dtype=np.float64)

            unkVec = T.concatenate((ligands, kfwd, rxnrates, nullRates, endo_activeEndo, sortF, kRec_kDeg, Rexpr, T.zeros(4, dtype=np.float64))) 

            Y_15 = self.dst15.calc(unkVec) # fitting the data based on dst15.calc for the given parameters
            Y_int = self.IL2Rb.calc(unkVec) # fitting the data based on dst.calc for the given parameters

            pm.Deterministic('Y_15', T.sum(T.square(Y_15)))
            pm.Deterministic('Y_int', T.sum(T.square(Y_int)))

            pm.Normal('fitD_15', sd=0.05, observed=Y_15) # TODO: Replace with experimental-derived stderr
            pm.Normal('fitD_int', sd=0.02, observed=Y_int)

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M

    def sampling(self):
        """This is the sampling that actually runs the model."""
        self.trace = pm.sample(init='ADVI', model=self.M)

    def profile(self):
        """ Profile the gradient calculation. """
        self.M.profile(pm.theanof.gradient(self.M.logpt, None)).summary()
