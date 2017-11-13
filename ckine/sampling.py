import numpy as np
from scipy.integrate import odeint
from model import dy_dt_IL2_wrapper
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

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
for ii in range (mat.shape[0]): # iterates through every combination of the arguments
    args = (1., mat[ii,0], mat[ii,1], mat[ii,2] )
    temp, d = odeint(dy_dt_IL2_wrapper, y0, ts, args, mxstep = 6000, full_output=True)
    if d['message'] == "Integration successful.": # only assign values to ys if there isn't an error message; all errors will still be 0
        ys[ii,:] = temp[1,:]
table = np.concatenate((mat, ys), 1) # puts the arguments to the left of the output data in a matrix
#print (table)
# fixed #18 

# now going to use PCA to make sense of the data
pca = PCA(n_components=10) # found percent of explained variance for each of the 13 components and the first 3 components each made up 33% of the explained variance
scores = pca.fit_transform(preprocessing.scale(table)) # this has my scores - now need to print/plot it
######## the following 5 lines found the % explained variance for each PC ############
exp_var = pca.explained_variance_ # this gives us the length of each eigenvalue in descending order
total_exp_var = sum(exp_var)
percent_exp_var = (exp_var * 100.) / total_exp_var
print (percent_exp_var)
print (sum(percent_exp_var))
print (pca.components_[0,:]) # this gives me the scaling for the loadings plot
print (scores[:,0])

