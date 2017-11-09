from model import dy_dt_IL2_wrapper
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pds

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

#print (IL2_activity_input([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.], 50., 2., 1., 1., 1.))

# this takes all the desired IL2 values we want to test and gives us the maximum activity value
# IL2 values pretty much ranged from 5 x 10**-4 to 500 nm with 8 points in between
def IL2_activity_values(y0, t, k4fwd, k5rev, k6rev):
    IL2s = np.logspace(-3.3, 2.7, 8) # 8 log-spaced values between our two endpoints
    activity = np.zeros(8)
    table = np.zeros((8,2))
    for ii in range (IL2s.shape[0]):
        activity[ii] = IL2_activity_input(y0, t, IL2s[ii], k4fwd, k5rev, k6rev)
        table[ii, 0] = IL2s[ii]
        table[ii, 1] = activity[ii]
    return table


def IL2_percent_activity(y0, t, k4fwd, k5rev, k6rev):
    values = IL2_activity_values(y0, t, k4fwd, k5rev, k6rev)
    maximum = np.amax(values[:,1], 0) # find the max value in the second column for all rows
    #print (maximum) 
    new_table = np.zeros((8,2))
    x = np.zeros((8,1))
    for ii in range (values.shape[0]):
        new_table[ii, 0] = values[ii, 0] # IL2 values in first column are the same
        new_table[ii, 1] = 100. * values[ii, 1] / maximum # activity values in second column are converted to percents relative to maximum
        x[ii] = math.log10(new_table[ii,0]) # changing the x values to the log10(nM) values that were in the published graph
    plt.rcParams.update({'font.size': 8})
    plt.xlabel("IL2 concentration (log(nm))")
    plt.ylabel("percent activation of pSTAT")
    plt.scatter(x[:], new_table[:,1])
    plt.show()
    return new_table
    


# now need to put IL2 extracted data into array format
## IL2 in IL2Ra- in YT-1 cells

def IL2_sum_mean_squared(y0, t, k4fwd, k5rev, k6rev):
    activity_table = IL2_percent_activity(y0, t, k4fwd, k5rev, k6rev) # generates output from percent activity function
    data = pds.read_csv("data\IL2_IL15_extracted_data.csv") # imports csv file into pandas array
    data.columns = ['IL2_conc','h9_2Ra-','IL2_2Ra-','IL15_2Ra-','IL15_15Ra_2Ra-', 'h9_2Ra+','IL2_2Ra+','IL15_2Ra+','IL15_15Ra_2Ra+' ] #change the names of the pandas columns in data
    print (data['IL2_2Ra-'])
    print (activity_table[:,1])
    pandas_activity = pds.DataFrame({'conc':[activity_table[:,0]], 'act':[activity_table[:,1]]}) # converts percent activity data into pandas file in order to allow for subtraction
    #differences_Ra- = pandas_activity['act'].sub(data['IL2_2Ra-'])
    print (type(pandas_activity['IL2_2Ra-'])) # this line is generating an error
    return True

IL2_sum_mean_squared([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.], 50., 1., 1., 1.)

