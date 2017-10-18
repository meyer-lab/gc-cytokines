import numpy as np
from scipy.integrate import odeint
from model import dy_dt_IL2_wrapper

# IL2 (third argument) will always be 1 nM and there will be 1000 of each receptor (first 3 elements of first argument (y))
# run odeint of dy_dt_IL2_wrapper([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.], ts, 1.,x,w,z) where x,w,z range from 10**-5 to 10**5 with increments by 10x
 
t = 50. # let's let the system run for 50 seconds
ts = np.linspace(0.0, t, 2)
y0 = np.array([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.])
x = np.logspace(-5, 5, num=11)
#print (x)
row = 0
ys = np.empty((11, 10))
#ys[1,] = odeint(dy_dt_IL2_wrapper, y0, ts, args, mxstep = 5000)
for ii, item in enumerate(x):
    args = (1., item, 10**-5, 10**-5)
    ys[ii,:] = odeint(dy_dt_IL2_wrapper, y0, ts, args, mxstep = 5000)[1,:]
print (ys)

#combinations module will make a list of things to loop through... function called combinations will take a bunch of vectors and give you all the combinations of them combined