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

print (IL2_activity_input([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.], 50, 2., 1., 1., 1.))

# this takes all the desired IL2 values we want to test and gives us the maximum activity value
def IL2_max_activity(IL2_start, IL2_stop, IL2_step):
    IL2_array = np.linspace(IL2_start, IL2_stop, IL2_step)
    activity_array = 
