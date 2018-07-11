"""
This file includes the classes and functions necessary to fit the IL2 model to the experimental data.
"""
from os.path import join, dirname, abspath
import pymc3 as pm
import theano
import theano.tensor as T
import numpy as np, pandas as pds
from .model import getTotalActiveSpecies, getSurfaceIL2RbSpecies
from .differencing_op import runCkineOp, runCkineKineticOp


class IL2Rb_trafficking:
    def __init__(self):
        path = dirname(abspath(__file__))
        # all of the IL2Rb trafficking data with IL2Ra+... first row contains headers... 9 columns and 8 rows... first column is time
        numpy_data = pds.read_csv(join(path, 'data/IL2Ra+_surface_IL2RB_datasets.csv')).values
        # all of the IL2Rb trafficking data with IL2Ra-... first row contains headers... 9 columns and 8 rows... first column is time
        numpy_data2 = pds.read_csv(join(path, "data/IL2Ra-_surface_IL2RB_datasets.csv")).values

        # times from experiment are hard-coded into this function
        self.ts = np.array([0., 2., 5., 15., 30., 60., 90.])

        # Condense to just IL2Rb
        self.condense = getSurfaceIL2RbSpecies()

        # Concatted data
        self.data = np.concatenate((numpy_data[:, 1], numpy_data[:, 5], numpy_data2[:, 1], numpy_data2[:, 5], numpy_data[:, 2], numpy_data[:, 6], numpy_data2[:, 2], numpy_data2[:, 6]))/10.

    def calc(self, unkVec):
        unkVecIL2RaMinus = T.set_subtensor(unkVec[18], 0.0) # Set IL2Ra to zero

        KineticOp = runCkineKineticOp(self.ts, self.condense)

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
        return T.concatenate((a, b, c, d, e, f, g, h)) / a[0] - self.data

# this takes all the desired IL2 values we want to test and gives us the maximum activity value
# IL2 values pretty much ranged from 5 x 10**-4 to 500 nm with 8 points in between
# need the theano decorator to get around the fact that there are if-else statements when running odeint but
# we don't necessarily know the values for the rxn rates when we call our model
class IL2_15_activity:
    def __init__(self):
        """This loads the experiment data and saves it as a member matrix and it also makes a vector of the IL15 concentrations that we are going to take care of."""
        path = dirname(abspath(__file__))
        data = pds.read_csv(join(path, "./data/IL2_IL15_extracted_data.csv")).values # imports csv file into pandas array
        self.cytokC = np.logspace(-3.3, 2.7, 8) # 8 log-spaced values between our two endpoints

        self.fit_data = np.concatenate((data[:, 7], data[:, 3], data[:, 6], data[:, 2])) / 100. #the IL15_IL2Ra- data is within the 4th column (index 3)

        self.activity = getTotalActiveSpecies().astype(np.float64)


    def calc(self, unkVec):
        """Simulate the experiment with IL15. It is making a list of promises which will be calculated and returned as output."""
        # Convert the vector of values to dicts

        # IL2Ra- cells have same IL15 activity, so we can just reuse same solution
        Op = runCkineOp(ts=np.array(500.))

        # Loop over concentrations of IL15
        actVec, _ = theano.map(fn=lambda x: T.dot(self.activity, Op(T.set_subtensor(unkVec[1], x))), sequences=[self.cytokC], name="IL15 loop")

        # Loop over concentrations of IL2
        actVecIL2, _ = theano.map(fn=lambda x: T.dot(self.activity, Op(T.set_subtensor(unkVec[0], x))), sequences=[self.cytokC])

        unkVecIL2RaMinus = T.set_subtensor(unkVec[18], 0.0) # Set IL2Ra to zero

        # Loop over concentrations of IL2, IL2Ra-/-
        actVecIL2RaMinus, _ = theano.map(fn=lambda x: T.dot(self.activity, Op(T.set_subtensor(unkVecIL2RaMinus[0], x))), sequences=[self.cytokC])

        # Normalize to the maximal activity, put together into one vector
        actCat = T.concatenate((actVec, actVec, actVecIL2, actVecIL2RaMinus))

        # value we're trying to minimize is the distance between the y-values on points of the graph that correspond to the same IL2 values
        return self.fit_data - (actCat / T.max(actCat))


class build_model:
    """Going to load the data from the CSV file at the very beginning of when build_model is called... needs to be separate member function to avoid uploading file thousands of times."""
    def __init__(self):
        self.dst15 = IL2_15_activity()
        self.IL2Rb = IL2Rb_trafficking()
        self.M = self.build()

    def build(self):
        """The PyMC model that incorporates Bayesian Statistics in order to store what the likelihood of the model is for a given point."""
        M = pm.Model()

        with M:
            kfwd = pm.Lognormal('kfwd', mu=np.log(0.00001), sd=0.1)
            rxnrates = pm.Lognormal('rxn', mu=np.log(0.1), sd=0.1, shape=8) # first 3 are IL2, second 5 are IL15, kfwd is first element (used in both 2&15)
            endo_activeEndo = pm.Lognormal('endo', mu=np.log(0.1), sd=0.1, shape=2)
            kRec_kDeg = pm.Lognormal('kRec_kDeg', mu=np.log(0.1), sd=0.1, shape=2)
            Rexpr = pm.Lognormal('IL2Raexpr', sd=0.1, shape=4) # Expression: IL2Ra, IL2Rb, gc, IL15Ra
            sortF = pm.Beta('sortF', alpha=20, beta=40, testval=0.333)*0.95

            ligands = T.zeros(4, dtype=np.float64)

            unkVec = T.concatenate((ligands, T.stack(kfwd), rxnrates, endo_activeEndo, T.stack(sortF), kRec_kDeg, Rexpr, T.zeros(2, dtype=np.float64)))

            Y_15 = self.dst15.calc(unkVec) # fitting the data based on dst15.calc for the given parameters
            Y_int = self.IL2Rb.calc(unkVec) # fitting the data based on dst.calc for the given parameters

            pm.Deterministic('Y_15', T.sum(T.square(Y_15)))
            pm.Deterministic('Y_int', T.sum(T.square(Y_int)))

            pm.Normal('fitD_15', sd=0.1, observed=Y_15) # TODO: Replace with experimental-derived stderr
            pm.Normal('fitD_int', sd=0.1, observed=Y_int)

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M

    def sampling(self):
        """This is the sampling that actually runs the model."""
        self.trace = pm.sample(init='ADVI', model=self.M)

    def profile(self):
        """ Profile the gradient calculation. """
        self.M.profile(pm.theanof.gradient(self.M.logpt, None)).summary()
