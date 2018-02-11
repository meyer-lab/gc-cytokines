"""
This file includes the classes and functions necessary to fit the IL15 model to the experimental data.
"""
import pymc3 as pm, theano.tensor as T, os
import numpy as np, pandas as pds
from .model import getTotalActiveCytokine, runCkine
from .differencing_op import centralDiff


#this takes the values of input parameters and calls odeint, then puts the odeint output into IL15_pSTAT_activity

def IL15_activity_input(IL15, rxnRates, trafRates):
    """Takes in the reaction rates, traficking rates, and the amount of IL15 that you want to simulate with, and it runs the model odeint. """
    rxnRates[1] = IL15

    ts = np.linspace(1., 500, 2)

    ys, retVal = runCkine(ts, rxnRates, trafRates)

    if retVal < 0:
        return -100

    return getTotalActiveCytokine(1, ys[1, :])

def IL15_convertRates(unkVec):
    """This takes in a vector of the values that we are fitting and it assigns them to the different reaction rates and ligand concentrations."""
    rxnRates = np.ones(17, dtype=np.float64)
    rxnRates[7:12] = unkVec[0:5] # k15rev, k17rev, k18rev, k22rev, k23rev
    rxnRates[0:4] = 0.0 # ligands

    tfR = np.zeros(11, dtype=np.float64)
    tfR[0:9] = unkVec[5:14]

    return (rxnRates, tfR)

# this takes all the desired IL15 values we want to test and gives us the maximum activity value
# IL15 values pretty much ranged from 5 x 10**-4 to 500 nm with 8 points in between
# need the theano decorator to get around the fact that there are if-else statements when running odeint but
#  we don't necessarily know the values for the rxn rates when we call our model

class IL15_sum_squared_dist:
    def __init__(self):
        """This loads the experiment data and saves it as a member matrix and it also makes a vector of the IL15 concentrations that we are going to take care of."""
        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(os.path.join(path, "./data/IL2_IL15_extracted_data.csv")) # imports csv file into pandas array
        self.numpy_data = data.as_matrix() #the IL15_IL2Ra- data is within the 4th column (index 3)
        self.IL15s = np.logspace(-3.3, 2.7, 8) # 8 log-spaced values between our two endpoints
        self.concs = len(self.IL15s)
        self.fit_data = np.concatenate((self.numpy_data[:, 7], self.numpy_data[:, 3]))

    def calc(self, unkVec):
        """Simulate the experiment with IL15. It is making a list of promises which will be calculated and returned as output."""
        # Convert the vector of values to dicts
        rxnRates, tfR = IL15_convertRates(unkVec)

        # IL2Ra- cells have same IL15 activity, so we can just reuse same solution
        actVec = np.zeros(self.concs, dtype=np.float64)

        # Loop over concentrations of IL15
        for ii, ILc in enumerate(self.IL15s):
            actVec[ii] = IL15_activity_input(ILc, rxnRates.copy(), tfR)

        # Normalize to the maximal activity, put together into one vector
        actVec = np.concatenate((actVec / np.max(actVec), actVec / np.max(actVec)))
        # value we're trying to minimize is the distance between the y-values on points of the graph that correspond to the same IL2 values
        return self.fit_data - actVec


class build_model:
    """Going to load the data from the CSV file at the very beginning of when build_model is called... needs to be separate member function to avoid uploading file thousands of times."""
    def __init__(self):
        self.dst = IL15_sum_squared_dist()
        self.M = self.build()

    def build(self):
        """The PyMC model incorporates Bayesian Statistics in order to store what the likelihood of the model is for a given point."""
        M = pm.Model()

        with M:
            rxnrates = pm.Lognormal('rxn', sd=1., shape=5, testval=[0.1, 0.1, 0.1,0.1,0.1]) # do we need to add a standard deviation? Yes, and they're all based on a lognormal scale
            endo_activeEndo = pm.Lognormal('endo', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            kRec_kDeg = pm.Lognormal('kRec_kDeg', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            Rexpr = pm.Lognormal('IL15Raexpr', sd=1., shape=4, testval=[1., 1., 1.,1.])
            sortF = pm.Beta('sortF', alpha=2, beta=7, testval=0.1)

            unkVec = T.concatenate((rxnrates, endo_activeEndo, T.stack(sortF), kRec_kDeg, Rexpr))

            Y = centralDiff(self.dst)(unkVec) # fitting the data based on dst.calc for the given parameters

            pm.Deterministic('Y', Y) # this line allows us to see the traceplots in read_fit_data.py... it lets us know if the fitting process is working

            pm.Normal('fitD', sd=0.1, observed=Y) # TODO: Find an empirical value for the SEM

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M

    def sampling(self):
        """This is the sampling that actually runs the model."""
        with self.M:
            try:
                self.trace = pm.sample()
            except ValueError:
                # Something went wrong, so print out the variables.
                print("Test point:")
                point = self.M.test_point
                logp = self.M.logp
                dlogp = self.M.dlogp()

                print(point)
                print(logp(point))
                print(dlogp(point))

                raise

    def profile(self):
        """ Profile the gradient calculation. """
        self.M.profile(pm.theanof.gradient(self.M.logpt, None)).summary()
