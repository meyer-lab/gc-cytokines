import numpy as np
from scipy.integrate import odeint
from model import dy_dt_IL2_wrapper

# IL2 (third argument) will always be 1 nM and there will be 1000 of each receptor (first 3 elements of first argument (y))
# run odeint of dy_dt_IL2_wrapper([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.], ts, 1.,x,w,z) where x,w,z range from 10**-5 to 10**5 with increments by 10x
 
t = 50. # let's let the system run for 50 seconds
ts = np.linspace(0.0, t, 2)
y0 = np.array([1000.,1000.,1000.,0.,0.,0.,0.,0.,0.,0.])
z = w = x = np.logspace(-5, 5, num=11) # creates a list with floats ranging from 10**-5 to 10**5
mat = np.array(np.meshgrid(w,x,z)).T.reshape(-1, 3)
#print (mat.shape[0]) # returns 1331 rows formed by this list
ys = np.zeros((1331, 10))
#print (mat[256,2])
for ii in range (mat.shape[0]):
    args = (1., mat[ii,0], mat[ii,1], mat[ii,2] )
    ys[ii,:] = odeint(dy_dt_IL2_wrapper, y0, ts, args, mxstep = 5000)[1,:] # will need to add full_output = True and make a conversion to a tuple (https://stackoverflow.com/questions/15260342/having-trouble-enabling-full-output-in-odeint) in order to get more details on the 3 error messages
print (ys)
