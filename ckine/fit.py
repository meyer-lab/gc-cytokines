from model import dy_dt_IL2_wrapper
from scipy.integrate import odeint
import numpy as np

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

print (IL2_activity_values([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.], 2., 1., 1., 1.))

def IL2_percent_activity(y0, t, k4fwd, k5rev, k6rev):
    values = IL2_activity_values(y0, t, k4fwd, k5rev, k6rev)
    maximum = np.amax(values[:,1], 0) #
    print (maximum) 
    new_table = np.zeros((8,2))
    for ii in range (values.shape[0]):
        new_table[ii, 0] = values[ii, 0]
        new_table[ii, 1] = 100. * values[ii, 1] / maximum
    return new_table
        
        
    
print (IL2_percent_activity([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.], 2., 1., 1., 1.))