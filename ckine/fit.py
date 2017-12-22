from .model import solveAutocrine, fullModel, getTotalActiveCytokine
from scipy.integrate import odeint
import numpy as np
import pandas as pds
from theano.compile.ops import as_op
import theano.tensor as T
import pymc3 as pm
import copy
import os
from concurrent.futures import ProcessPoolExecutor

global pool
pool = ProcessPoolExecutor()

# this takes the values of input parameters and calls odeint, then puts the odeint output into IL2_pSTAT_activity
def IL2_activity_input(y0, IL2, rxnRates, trafRates):
    rxnRates['IL2'] = IL2
    ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates)
    ts = np.linspace(0., 500, 2)

    ys, infodict = odeint(ddfunc, y0, ts, mxstep=6000, full_output=True)

    if infodict['tcur'] < np.max(ts):
        print(IL2)
        return -100

    return getTotalActiveCytokine(0, ys[1, :])

# this takes all the desired IL2 values we want to test and gives us the maximum activity value
# IL2 values pretty much ranged from 5 x 10**-4 to 500 nm with 8 points in between
# need the theano decorator to get around the fact that there are if-else statements when running odeint but
#  we don't necessarily know the values for the rxn rates when we call our model
@as_op(itypes=[T.dvector], otypes=[T.dmatrix])
def IL2_activity_values(unkVec):
    IL2s = np.logspace(-3.3, 2.7, 8) # 8 log-spaced values between our two endpoints
    table = np.zeros((8, 2))
    output = list()

    rxnRates = dict({'IL15':0.0, 'IL7':0.0, 'IL9':0.0, 'k15rev':1.0, 'k17rev':1.0, 'k18rev':1.0,
                     'k22rev':1.0, 'k23rev':1.0, 'k26rev':1.0, 'k27rev':1.0, 'k29rev':1.0, 'k30rev':1.0, 'k31rev':1.0})
    rxnRates['kfwd'], rxnRates['k5rev'], rxnRates['k6rev'] = unkVec[0:3]

    trafRates = dict()
    trafRates['endo'] = unkVec[3]
    trafRates['activeEndo'] = unkVec[6]
    trafRates['sortF'] = 0.1
    trafRates['activeSortF'] = 1.0
    trafRates['kRec'] = unkVec[4]
    trafRates['kDeg'] = unkVec[5]
    trafRates['exprV'] = np.array([unkVec[7], unkVec[8], unkVec[9], 0.0, 0.0, 0.0], dtype=np.float64)

    yAutocrine = solveAutocrine(rxnRates, trafRates)

    global pool

    if 'pool' in globals():
        for ii, ILc in enumerate(IL2s):
            output.append(pool.submit(IL2_activity_input, yAutocrine, ILc, copy.deepcopy(rxnRates), trafRates))

        for ii, item in enumerate(output):
            table[ii, 1] = item.result()
    else:
        print("Note: Not running parallel.")
        for ii, ILc in enumerate(IL2s):
            table[ii, 1] = IL2_activity_input(yAutocrine, ILc, copy.deepcopy(rxnRates), trafRates)
    
    table[:, 0] = IL2s

    return table


def IL2_percent_activity(unkVec):
    values = IL2_activity_values(unkVec)
    maximum = T.max(values[:,1], 0) # find the max value in the second column for all rows

    new_table = T.stack((values[:, 0], 100. * values[:, 1] / maximum), axis=1) # IL2 values in first column are the same
    # activity values in second column are converted to percents relative to maximum
    
    return new_table


class IL2_sum_squared_dist:
    
    def load(self):
        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(os.path.join(path, "./data/IL2_IL15_extracted_data.csv")) # imports csv file into pandas array
        self.numpy_data = data.as_matrix() #the IL2_IL2Ra- data is within the 3rd column (index 2)
        
    def calc(self, unkVec):
        activity_table = IL2_percent_activity(unkVec)
        diff_data = self.numpy_data[:,6] - activity_table[:,1] # value we're trying to minimize is the distance between the y-values on points of the graph that correspond to the same IL2 values
        return np.squeeze(diff_data)


class build_model:
    
    # going to load the data from the CSV file at the very beginning of when build_model is called... needs to be separate member function to avoid uploading file thousands of times
    def __init__(self):
        self.dst = IL2_sum_squared_dist()
        self.dst.load() 
    
    def build(self):
        self.M = pm.Model()
        
        with self.M:
            rxnrates = pm.Lognormal('rxn', mu=0, sd=3, shape=3) # do we need to add a standard deviation? Yes, and they're all based on a lognormal scale
            Rexpr = pm.Lognormal('trafR', mu=1, sd=2, shape=4)
            trafR = pm.Lognormal('IL2Raexpr', mu=-1, sd=2, shape=3)

            unkVec = T.concatenate((rxnrates, trafR, Rexpr))
            
            Y = self.dst.calc(unkVec) # fitting the data based on dst.calc for the given parameters
            
            pm.Deterministic('Y', Y) # this line allows us to see the traceplots in read_fit_data.py... it lets us know if the fitting process is working

            pm.Normal('fitD', mu=0, sd=T.std(Y), observed=Y)

            pm.Deterministic('logp', self.M.logpt)
    
    def sampling(self):
        with self.M:
            start = pm.find_MAP()
            step = pm.Metropolis()
            self.trace = pm.sample(5000, step, start=start) # 5000 represents the number of steps taken in the walking process
