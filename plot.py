import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from ckine.model import dy_dt

# this plot is incomplete as it is missing labels for each element of ys on the plot
def plot(t, IL2Ra, IL2Rb, gc, IL2, k1fwd, k4fwd, k5rev, k6rev, k10rev, k11rev):
    ts = np.linspace(0.0, t, 100000)
    # in this system we want no initial IL2, thus y0[0:2] can be set as inputes but y0[3:9] = 0
    y0 = [IL2Ra, IL2Rb, gc, 0., 0., 0., 0., 0., 0., 0.]
    args = (IL2, k1fwd, k4fwd, k5rev, k6rev, k10rev, k11rev)
    ys = odeint(dy_dt, y0, ts, args, mxstep = 5000)
    # Plot the numerical solution
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel("t")
    plt.ylabel("y")
    plt.plot(ts, ys);

# example of calling the plot with randomnly selected arguments    
plot(10, 3., 1., 0.5, 0.2, 0.5, 0.9, 1.0, 1.8, 1.2, 0.8)