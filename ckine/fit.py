"""
This file includes the classes and functions necessary to fit the IL2 model to the experimental data.
"""
import pymc3 as pm, theano.tensor as T, os
import numpy as np, pandas as pds
from .model import getTotalActiveCytokine, runCkine
from .differencing_op import centralDiff


def IL_activity_input(IL, rxnRates, trafRates, cytokineIDX):
    """Takes in the reaction rates, traficking rates, and the amount of IL2 that you want to simulate with, and it runs the model."""
    rxnRates[cytokineIDX] = IL

    ts = np.array([500.])

    ys, retVal = runCkine(ts, rxnRates, trafRates)

    if retVal < 0:
        return -100

    return getTotalActiveCytokine(cytokineIDX, ys[0, :])


def convertRates(unkVec):
    """This takes in a vector of the values that we are fitting and it assigns them to the different reaction rates and ligand concentrations."""
    rxnRates = np.ones(15, dtype=np.float64)
    rxnRates[4:12] = unkVec[0:8] # kfwd, k5rev, k6rev, k15rev, k17rev, k18rev, k22rev, k23rev
    rxnRates[0:4] = 0.0 # ligands

    tfR = np.zeros(11, dtype=np.float64)
    tfR[0:9] = unkVec[8:17] # last 4 are expr of IL2Ra, IL2Rb, gc, IL15Ra

    return (rxnRates, tfR)


def surf_IL2Rb(rxnRates, trafRates, IL2_conc):
    # times from experiment are hard-coded into this function
    ts = np.array(([0.01, 2., 5., 15., 30., 60., 90.]))

    rxnRates[0] = IL2_conc # the concentration of IL2 is rxnRates[0]

    ys, retVal = runCkine(ts, rxnRates, trafRates)

    if retVal < 0:
        return -100

    sIL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel

    return 10. * (sIL2Rb / sIL2Rb[0]) # % sIL2Rb relative to initial amount


class IL2Rb_trafficking:
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(os.path.join(path, 'data/IL2Ra+_surface_IL2RB_datasets.csv')) # imports csv file into pandas array
        self.numpy_data = data.as_matrix() # all of the IL2Rb trafficking data with IL2Ra+... first row contains headers... 9 columns and 8 rows... first column is time
        data2 = pds.read_csv(os.path.join(path, "data/IL2Ra-_surface_IL2RB_datasets.csv"))
        self.numpy_data2 = data2.as_matrix() # all of the IL2Rb trafficking data with IL2Ra-... first row contains headers... 9 columns and 8 rows... first column is time

        self.concs = 14

    def calc(self, unkVec):
        # Convert the vector of values to dicts
        rxnRates, tfR = convertRates(unkVec)

        # IL2Ra- cells
        tfR2 = tfR.copy()
        tfR2[5] = 0.0

        diff1 = surf_IL2Rb(rxnRates, tfR, 1) - self.numpy_data[:, 1] # the second column of numpy_data has all the 1nM IL2Ra+ data
        diff2 = surf_IL2Rb(rxnRates, tfR, 500) - self.numpy_data[:, 5] # the sixth column of numpy_data has all the 500 nM IL2Ra+ data
        diff3 = surf_IL2Rb(rxnRates, tfR2, 1) - self.numpy_data2[:, 1] # the second column of numpy_data2 has all the 1nM IL2Ra- data
        diff4 = surf_IL2Rb(rxnRates, tfR2, 500) - self.numpy_data2[:, 5] # the sixth column of numpy_data2 has all the 500 nM IL2Ra- data

        all_diffs = np.concatenate((diff1, diff2, diff3, diff4))

        return all_diffs


# this takes all the desired IL2 values we want to test and gives us the maximum activity value
# IL2 values pretty much ranged from 5 x 10**-4 to 500 nm with 8 points in between
# need the theano decorator to get around the fact that there are if-else statements when running odeint but
# we don't necessarily know the values for the rxn rates when we call our model
class IL2_sum_squared_dist:
    def __init__(self):
        """This loads the experiment data and saves it as a member matrix and it also makes a vector of the IL2 concentrations that we are going to take care of."""
        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(os.path.join(path, "./data/IL2_IL15_extracted_data.csv")) # imports csv file into pandas array
        self.numpy_data = data.as_matrix() #the IL2_IL2Ra- data is within the 3rd column (index 2)
        self.IL2s = np.logspace(-3.3, 2.7, 8) # 8 log-spaced values between our two endpoints
        self.concs = len(self.IL2s)
        self.fit_data = np.concatenate((self.numpy_data[:, 6], self.numpy_data[:, 2]))

    def calc(self, unkVec):
        """Simulate the 2 experiments: one w/ IL2Ra and one without it. It is making a list of promises which will be calculated and returned as output."""
        # Convert the vector of values to dicts
        rxnRates, tfR = convertRates(unkVec)

        # IL2Ra- cells
        tfR2 = tfR.copy()
        tfR2[5] = 0.0

        actVec = np.zeros(self.concs, dtype=np.float64)
        actVec2 = np.zeros(self.concs, dtype=np.float64)

        # Loop over concentrations of IL2
        for ii, ILc in enumerate(self.IL2s):
            actVec[ii] = IL_activity_input(ILc, rxnRates.copy(), tfR, 0)
            actVec2[ii] = IL_activity_input(ILc, rxnRates.copy(), tfR2, 0)

        # Normalize to the maximal activity, put together into one vector
        actVec = np.concatenate((actVec / np.max(actVec), actVec2 / np.max(actVec2)))

        # value we're trying to minimize is the distance between the y-values on points of the graph that correspond to the same IL2 values
        return self.fit_data - actVec
    
    
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
        rxnRates, tfR = convertRates(unkVec)

        # IL2Ra- cells have same IL15 activity, so we can just reuse same solution
        actVec = np.zeros(self.concs, dtype=np.float64)

        # Loop over concentrations of IL15
        for ii, ILc in enumerate(self.IL15s):
            actVec[ii] = IL_activity_input(ILc, rxnRates.copy(), tfR, 1)

        # Normalize to the maximal activity, put together into one vector
        actVec = np.concatenate((actVec / np.max(actVec), actVec / np.max(actVec)))
        # value we're trying to minimize is the distance between the y-values on points of the graph that correspond to the same IL2 values
        return self.fit_data - actVec
    

class build_model:
    """Going to load the data from the CSV file at the very beginning of when build_model is called... needs to be separate member function to avoid uploading file thousands of times."""
    def __init__(self):
        self.dst2 = IL2_sum_squared_dist()
        self.dst15 = IL15_sum_squared_dist()
        self.IL2Rb = IL2Rb_trafficking()
        self.M = self.build()

    def build(self):
        """The PyMC model that incorporates Bayesian Statistics in order to store what the likelihood of the model is for a given point."""
        M = pm.Model()

        with M:
            rxnrates = pm.Lognormal('rxn', sd=1., shape=8, testval=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # first 3 are IL2, second 5 are IL15, kfwd is first element (used in both 2&15)
            endo_activeEndo = pm.Lognormal('endo', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            kRec_kDeg = pm.Lognormal('kRec_kDeg', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            Rexpr = pm.Lognormal('IL2Raexpr', sd=1., shape=4, testval=[1., 1., 1., 1.]) # Expression: IL2Ra, IL2Rb, gc, IL15Ra
            sortF = pm.Beta('sortF', alpha=2, beta=7, testval=0.1)

            unkVec = T.concatenate((rxnrates, endo_activeEndo, T.stack(sortF), kRec_kDeg, Rexpr))

            Y_2 = centralDiff(self.dst2)(unkVec) # fitting the data based on dst2.calc for the given parameters
            Y_15 = centralDiff(self.dst15)(unkVec) # fitting the data based on dst15.calc for the given parameters
            Y_int = centralDiff(self.IL2Rb)(unkVec) # fitting the data based on dst.calc for the given parameters

            pm.Deterministic('Y_2', T.sum(T.square(Y_2))) # Save the sum of squared error to see if we're fitting
            pm.Deterministic('Y_15', T.sum(T.square(Y_15)))
            pm.Deterministic('Y_int', T.sum(T.square(Y_int)))

            pm.Normal('fitD_2', sd=T.std(Y_2), observed=Y_2)
            pm.Normal('fitD_15', sd=T.std(Y_15), observed=Y_15)
            pm.Normal('fitD_int', sd=T.std(Y_int), observed=Y_int)

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M

    def sampling(self):
        """This is the sampling that actually runs the model."""
        self.trace = pm.sample(init='advi', model=self.M)

    def fit_ADVI(self):
        with self.M:
            approx = pm.fit(40000, method='fullrank_advi')
            self.trace = approx.sample()

    def profile(self):
        """ Profile the gradient calculation. """
        self.M.profile(pm.theanof.gradient(self.M.logpt, None)).summary()
