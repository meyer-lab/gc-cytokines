import numpy as np
from scipy.integrate import odeint
from model import dy_dt_IL15_wrapper
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


t = 50. # let's let the system run for 50 seconds
ts = np.linspace(0.0, t, 2)
y0 = np.array([1000.,1000.,1000., 0., 0., 0., 0., 0., 0., 0.])
a = b = c = d = e = f = np.logspace(-2, 2, num=3)
mat = np.array(np.meshgrid(a,b,c,d,e,f)).T.reshape(-1, 6)
ys = np.zeros(729,10)
for i in range (len(mat)):
    args = (1., mat[i,0], mat[i,1], mat[i,2], mat[i,3], mat[i,4], mat[i,5])
    temp, d = odeint(dy_dt_IL15_wrapper, y0, ts, args, mxstep = 6000, full_output=True)
    ys[i,:] = temp[1,:]
table = np.concatenate((mat, ys), 1) # puts the arguments to the left of the output data in a matrix
#print (table)


# now going to use PCA to make sense of the data
pca = PCA(n_components=10)
scores = pca.fit_transform(preprocessing.scale(table))
exp_var = pca.explained_variance_ # this gives us the length of each eigenvalue in descending order
total_exp_var = sum(exp_var)
percent_exp_var = (exp_var * 100.) / total_exp_var
print (percent_exp_var)
print (sum(percent_exp_var))
#print (pca.components_[0,:]) # this gives the scaling for the loadings plot
#print (scores[:,0])





