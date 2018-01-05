from .model import solveAutocrine, fullModel, getTotalActiveCytokine, __active_species_IDX, printModel
from scipy.integrate import odeint
import numpy as np, pandas as pds
from .differencing_op import centralDiff
import pymc3 as pm, theano.tensor as T, os


# this takes the values of input parameters and calls odeint, then puts the odeint output into IL2_pSTAT_activity
def IL2_activity_input(y0, IL2, rxnRates, trafRates):
    rxnRates[0] = IL2

    ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
    ts = np.linspace(0., 500, 2)

    ys, infodict = odeint(ddfunc, y0, ts, mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)

    if infodict['tcur'] < np.max(ts):
        print("IL2 conc: " + str(IL2))
        printModel(rxnRates, trafRates)
        print(infodict)
        return -100

    return getTotalActiveCytokine(0, ys[1, :])


def IL2_convertRates(unkVec):
    rxnRates = np.ones(17, dtype=np.float64)
    rxnRates[4:7] = unkVec[0:3] # kfwd, k5rev, k6rev
    rxnRates[0:4] = 0.0 # ligands

    tfR = np.zeros(11, dtype=np.float64)
    tfR[0:8] = unkVec[3:11]

    return (rxnRates, tfR)


# this takes all the desired IL2 values we want to test and gives us the maximum activity value
# IL2 values pretty much ranged from 5 x 10**-4 to 500 nm with 8 points in between
# need the theano decorator to get around the fact that there are if-else statements when running odeint but
#  we don't necessarily know the values for the rxn rates when we call our model
class IL2_sum_squared_dist:
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(os.path.join(path, "./data/IL2_IL15_extracted_data.csv")) # imports csv file into pandas array
        self.numpy_data = data.as_matrix() #the IL2_IL2Ra- data is within the 3rd column (index 2)
        self.IL2s = np.logspace(-3.3, 2.7, 8) # 8 log-spaced values between our two endpoints
        self.concs = len(self.IL2s)
        self.fit_data = np.concatenate((self.numpy_data[:, 6], self.numpy_data[:, 2]))

    def calc_schedule(self, unkVec, pool):
        # Convert the vector of values to dicts
        rxnRates, tfR = IL2_convertRates(unkVec)

        # IL2Ra- cells
        tfR2 = tfR.copy()
        tfR2[5] = 0.0

        # Find autocrine state
        yAutocrine = solveAutocrine(tfR)
        yAutocrine2 = solveAutocrine(tfR2)

        # Loop over concentrations of IL2
        output = list()
        output2 = list()

        for _, ILc in enumerate(self.IL2s):
            output.append(pool.submit(IL2_activity_input, yAutocrine, ILc, rxnRates.copy(), tfR))

        for _, ILc in enumerate(self.IL2s):
            output2.append(pool.submit(IL2_activity_input, yAutocrine2, ILc, rxnRates.copy(), tfR2))

        return (output, output2)

    def calc_reduce(self, inT):
        output, output2 = inT

        actVec = np.fromiter((item.result() for item in output), np.float64, count=self.concs)
        actVec2 = np.fromiter((item.result() for item in output2), np.float64, count=self.concs)

        # Normalize to the maximal activity, put together into one vector
        actVec = np.concatenate((actVec / np.max(actVec), actVec2 / np.max(actVec2)))

        # value we're trying to minimize is the distance between the y-values on points of the graph that correspond to the same IL2 values
        return self.fit_data - actVec

    def calc(self, unkVec, pool):
        """ Just get the solution in one pass. """
        inT = self.calc_schedule(unkVec, pool)
        return self.calc_reduce(inT)


class build_model:
    
    # going to load the data from the CSV file at the very beginning of when build_model is called... needs to be separate member function to avoid uploading file thousands of times
    def __init__(self):
        self.dst = IL2_sum_squared_dist()
        self.M = self.build()
    
    def build(self):
        M = pm.Model()

        with M:
            rxnrates = pm.Lognormal('rxn', sd=1., shape=3, testval=[0.1, 0.1, 0.1]) # do we need to add a standard deviation? Yes, and they're all based on a lognormal scale
            endo_activeEndo = pm.Lognormal('endo', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            kRec_kDeg = pm.Lognormal('kRec_kDeg', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            Rexpr = pm.Lognormal('IL2Raexpr', sd=1., shape=3, testval=[1., 1., 1.])
            sortF = pm.Beta('sortF', alpha=2, beta=7, testval=0.1)

            unkVec = T.concatenate((rxnrates, endo_activeEndo, T.stack(sortF), kRec_kDeg, Rexpr))
            
            Y = centralDiff(self.dst)(unkVec) # fitting the data based on dst.calc for the given parameters
            
            pm.Deterministic('Y', Y) # this line allows us to see the traceplots in read_fit_data.py... it lets us know if the fitting process is working

            pm.Normal('fitD', sd=0.1, observed=Y) # TODO: Find an empirical value for the SEM

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M
    
    def sampling(self):
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
