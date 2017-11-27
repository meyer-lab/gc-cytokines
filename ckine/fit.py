from model import dy_dt_IL2_wrapper
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pds
from theano.compile.ops import as_op
import theano.tensor as T
import pymc3 as pm
import pickle as pk
import bz2
import concurrent.futures


pool = concurrent.futures.ProcessPoolExecutor()

# this just takes the output of odeint (y values) and determines pSTAT activity
def IL2_pSTAT_activity(ys):
    # pSTAT activation is based on sum of IL2_IL2Rb_gc and IL2_IL2Ra_IL2Rb_gc at long time points
    activity = ys[1, 8] + ys[1, 9]
    return activity

# this takes the values of input parameters and calls odeint, then puts the odeint output into IL2_pSTAT_activity
def IL2_activity_input(y0, t, IL2, k4fwd, k5rev, k6rev):
    args = (IL2, k4fwd, k5rev, k6rev)
    ts = np.linspace(0., t, 2)
    ys = odeint(dy_dt_IL2_wrapper, y0, ts, args, mxstep = 6000)
    act = IL2_pSTAT_activity(ys)
    return act

# this takes all the desired IL2 values we want to test and gives us the maximum activity value
# IL2 values pretty much ranged from 5 x 10**-4 to 500 nm with 8 points in between
@as_op(itypes=[T.dscalar, T.dscalar, T.dscalar], otypes=[T.dmatrix])
def IL2_activity_values(k4fwd, k5rev, k6rev):
    y0 = np.array([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.])
    t = 50.
    IL2s = np.logspace(-3.3, 2.7, 8) # 8 log-spaced values between our two endpoints
    table = np.zeros((8, 2))
    output = list()

    for ii in range(len(IL2s)):
        output.append(pool.submit(IL2_activity_input, y0, t, IL2s[ii], k4fwd, k5rev, k6rev))

    for ii in range(len(IL2s)):
        table[ii, 1] = output[ii].result()
    
    table[:, 0] = IL2s

    return table



def IL2_percent_activity(k4fwd, k5rev, k6rev):
    values = IL2_activity_values(k4fwd, k5rev, k6rev)
    maximum = T.max(values[:,1], 0) # find the max value in the second column for all rows

    new_table = T.stack((values[:, 0], 100. * values[:, 1] / maximum), axis=1) # IL2 values in first column are the same
    # activity values in second column are converted to percents relative to maximum
    
    return new_table


def plot_IL2_percent_activity(y0, t, k4fwd, k5rev, k6rev):
    new_table = IL2_percent_activity(k4fwd, k5rev, k6rev)

    x = math.log10(new_table[:, 0]) # changing the x values to the log10(nM) values that were in the published graph

    plt.rcParams.update({'font.size': 8})
    plt.xlabel("IL2 concentration (log(nm))")
    plt.ylabel("percent activation of pSTAT")
    plt.scatter(x[:], new_table[:,1])
    plt.show()




class IL2_sum_squared_dist:
    
    def load(self):
        data = pds.read_csv("./data/IL2_IL15_extracted_data.csv") # imports csv file into pandas array
        self.numpy_data = data.as_matrix() #the IL2_IL2Ra- data is within the 3rd column (index 2)
        
    def calc(self, k4fwd, k5rev, k6rev):
        activity_table = IL2_percent_activity(k4fwd, k5rev, k6rev)
        diff_data = self.numpy_data[:,6] - activity_table[:,1]
        return np.squeeze(diff_data)
        
def store_data(class_name, fit_results):
    pk.dump(class_name, bz2.BZ2File(fit_results + '.pkl', 'wb'))

class build_model:
    
    def __init__(self):
        self.dst = IL2_sum_squared_dist()
        self.dst.load()
    
    def build(self):   
        self.M = pm.Model()
        
        with self.M:
            k4fwd = pm.Lognormal('k4fwd', mu=0, sd=3) # do we need to add a standard deviation? Yes, and they're all based on a lognormal scale
            k5rev = pm.Lognormal('k5rev', mu=0, sd=3)
            k6rev = pm.Lognormal('k6rev', mu=0, sd=3)
            
            Y = self.dst.calc(k4fwd, k5rev, k6rev)
            
            Y_obs = pm.Normal('fitD', mu=0, sd=T.std(Y), observed=Y)
        
        return Y_obs
    
    def sampling(self):
        with self.M:
            start = pm.find_MAP()
            step = pm.Metropolis()
            self.trace = pm.sample(5000, step, start=start) # original value should be 5 to shorten time
            
        
_ = plt.hist(build_model.trace['k4fwd'],100) # no longer need the 'self' because I am executing this line outside of the class

store_data(build_model, "model_results")