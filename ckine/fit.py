"""
This file includes the classes and functions necessary to fit the IL2 model to the experimental data.
"""
import pymc3 as pm, theano.tensor as T, os
import numpy as np, pandas as pds
from .model import getTotalActiveCytokine, printModel, runCkine
from .differencing_op import centralDiff


def IL2_activity_input(IL2, rxnRates, trafRates):
    """Takes in the reaction rates, traficking rates, and the amount of IL2 that you want to simulate with, and it runs the model."""
    rxnRates[0] = IL2

    ts = np.linspace(1., 500, 2)

    ys, retVal = runCkine(ts, rxnRates, trafRates)

    if retVal < 0:
        return -100

    return getTotalActiveCytokine(0, ys[1, :])


def IL2_convertRates(unkVec):
    """This takes in a vector of the values that we are fitting and it assigns them to the different reaction rates and ligand concentrations."""
    rxnRates = np.ones(17, dtype=np.float64)
    rxnRates[4:7] = unkVec[0:3] # kfwd, k5rev, k6rev
    rxnRates[0:4] = 0.0 # ligands

    tfR = np.zeros(11, dtype=np.float64)
    tfR[0:8] = unkVec[3:11]

    return (rxnRates, tfR)


def surf_IL2Rb(rxnRates, trafRates, IL2_conc):
    # times from experiment are hard-coded into this function
    ts = np.array(([0.01, 2., 5., 15., 30., 60., 90.]))

    rxnRates[0] = IL2_conc # the concentration of IL2 is rxnRates[0]

    ys, retVal = runCkine(ts, rxnRates, trafRates)

    if retVal < 0:
        print("Model run failed")
        printModel(rxnRates, trafRates)
        return -100

    surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points

    percent_surface_IL2Rb = 10. * (surface_IL2Rb / surface_IL2Rb[0]) # percent of surface IL2Rb is relative to the initial amount of receptor

    return percent_surface_IL2Rb


class IL2Rb_trafficking:
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(os.path.join(path, 'data/IL2Ra+_surface_IL2RB_datasets.csv')) # imports csv file into pandas array
        self.numpy_data = data.as_matrix() # all of the IL2Rb trafficking data with IL2Ra+... first row contains headers... 9 columns and 8 rows... first column is time
        data2 = pds.read_csv(os.path.join(path, "data/IL2Ra-_surface_IL2RB_datasets.csv"))
        self.numpy_data2 = data2.as_matrix() # all of the IL2Rb trafficking data with IL2Ra-... first row contains headers... 9 columns and 8 rows... first column is time

        self.concs = 14

    def calc_schedule(self, unkVec, pool):
        # Convert the vector of values to dicts
        rxnRates, tfR = IL2_convertRates(unkVec)

        # IL2Ra- cells
        tfR2 = tfR.copy()
        tfR2[5] = 0.0

        # Loop over concentrations of IL2
        output = list()

        output.append(pool.submit(surf_IL2Rb, rxnRates, tfR, 1)) # handle the IL2Ra+ and 1nM case
        output.append(pool.submit(surf_IL2Rb, rxnRates, tfR, 500)) # handle the IL2Ra+ and 500nM case
        output.append(pool.submit(surf_IL2Rb, rxnRates, tfR2, 1)) # handle the IL2Ra- and 1nM case
        output.append(pool.submit(surf_IL2Rb, rxnRates, tfR2, 500)) # handle the IL2Ra- and 500nM case

        return output


    def calc_reduce(self, inT):
        actVec = list(item.result() for item in inT)

        # actVec[0] represents the IL2Ra+ and 1nM case
        # actVec[1] represents the IL2Ra+ and 500nM case
        # actVec[2] represents the IL2Ra- and 1nM case
        # actVec[3] represents the IL2Ra- and 500nM case
        diff = actVec[0] - self.numpy_data[:, 1] # the second column of numpy_data has all the 1nM IL2Ra= data
        diff2 = actVec[1] - self.numpy_data[:, 5] # the sixth column of numpy_data has all the 500 nM IL2Ra+ data
        diff3 = actVec[2] - self.numpy_data2[:, 1] # the second column of numpy_data2 has all the 1nM IL2Ra- data
        diff4 = actVec[3] - self.numpy_data2[:, 5] # the sixth column of numpy_data2 has all the 500 nM IL2Ra- data

        all_diffs = np.concatenate((diff, diff2, diff3, diff4))

        return all_diffs
    def calc(self, unkVec, pool):
        """ Just get the solution in one pass. """
        inT = self.calc_schedule(unkVec, pool)
        return self.calc_reduce(inT)



# this takes all the desired IL2 values we want to test and gives us the maximum activity value
# IL2 values pretty much ranged from 5 x 10**-4 to 500 nm with 8 points in between
# need the theano decorator to get around the fact that there are if-else statements when running odeint but
#  we don't necessarily know the values for the rxn rates when we call our model
class IL2_sum_squared_dist:
    def __init__(self):
        """This loads the experiment data and saves it as a member matrix and it also makes a vector of the IL2 concentrations that we are going to take care of."""
        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(os.path.join(path, "./data/IL2_IL15_extracted_data.csv")) # imports csv file into pandas array
        self.numpy_data = data.as_matrix() #the IL2_IL2Ra- data is within the 3rd column (index 2)
        self.IL2s = np.logspace(-3.3, 2.7, 8) # 8 log-spaced values between our two endpoints
        self.concs = len(self.IL2s)
        self.fit_data = np.concatenate((self.numpy_data[:, 6], self.numpy_data[:, 2]))

    def calc_schedule(self, unkVec, pool):
        """Simulate the 2 experiments: one w/ IL2Ra and one without it. It is making a list of promises which will be calculated and returned as output."""
        # Convert the vector of values to dicts
        rxnRates, tfR = IL2_convertRates(unkVec)

        # IL2Ra- cells
        tfR2 = tfR.copy()
        tfR2[5] = 0.0

        # Loop over concentrations of IL2
        output = list()
        output2 = list()

        for _, ILc in enumerate(self.IL2s):
            output.append(pool.submit(IL2_activity_input, ILc, rxnRates.copy(), tfR))

        for _, ILc in enumerate(self.IL2s):
            output2.append(pool.submit(IL2_activity_input, ILc, rxnRates.copy(), tfR2))

        return (output, output2)

    def calc_reduce(self, inT):
        """After getting all of the promises first, calc_reduce is going to convert all those promises into actual values and return the difference between the measurements and the simulation."""
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
    """Going to load the data from the CSV file at the very beginning of when build_model is called... needs to be separate member function to avoid uploading file thousands of times."""
    def __init__(self):
        self.dst = IL2_sum_squared_dist()
        self.IL2Rb = IL2Rb_trafficking()
        self.M = self.build()

    def build(self):
        """The PyMC model that incorporates Bayesian Statistics in order to store what the likelihood of the model is for a given point."""
        M = pm.Model()

        with M:
            rxnrates = pm.Lognormal('rxn', sd=1., shape=3, testval=[0.1, 0.1, 0.1])
            endo_activeEndo = pm.Lognormal('endo', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            kRec_kDeg = pm.Lognormal('kRec_kDeg', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            Rexpr = pm.Lognormal('IL2Raexpr', sd=1., shape=3, testval=[1., 1., 1.])
            sortF = pm.Beta('sortF', alpha=2, beta=7, testval=0.1)

            unkVec = T.concatenate((rxnrates, endo_activeEndo, T.stack(sortF), kRec_kDeg, Rexpr))

            Y = centralDiff(self.dst)(unkVec) # fitting the data based on dst.calc for the given parameters

            Y_int = centralDiff(self.IL2Rb)(unkVec) # fitting the data based on dst.calc for the given parameters

            pm.Deterministic('Y', Y) # this line allows us to see the traceplots in read_fit_data.py... it lets us know if the fitting process is working

            pm.Deterministic('Y_int', Y_int)

            pm.Normal('fitD', sd=0.1, observed=Y) # TODO: Find an empirical value for the SEM
            pm.Normal('fitD_int', sd=0.1, observed=Y_int) # TODO: Find an empirical value for the SEM

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M

    def sampling(self):
        """This is the sampling that actually runs the model."""
        with self.M:
            try:
                self.trace = pm.sample(init='advi')
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
