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
#print (pca.components_[0,:]) # this gives me the scaling for the loadings plot
#print (scores[:,0])

# generate scores plot of PC1 vs PC2
plt.rcParams.update({'font.size': 8})
plt.xlabel("PC 1 scores (33%)")
plt.ylabel("PC 2 scores (33%)")
plt.scatter(scores[:,0], scores[:,1])
plt.show()

#generate scores plot of PC1 vs PC3
plt.rcParams.update({'font.size': 8})
plt.xlabel("PC 1 scores (33%)")
plt.ylabel("PC 3 scores (33%)")
plt.scatter(scores[:,0], scores[:,2])
plt.show() # take out both of the plt.show() lines if you want all the data to be on the same plot and color coded

# setting up variables to simplify the creation of loadings plots
comp = pca.components_
# print (comp)
pc1 = comp[0,:]
pc2 = comp[1,:]
pc3 = comp[2,:]
# print (pc1)

# generate loadings plot of PC1 vs PC2

# first need to do some set-up for my annotations
fig = plt.figure()
ax = fig.add_subplot(111)
# plot all the points from PC1 and PC2 in a scatter plot
plt.rcParams.update({'font.size': 8})
plt.xlabel("PC 1 loadings (33%)")
plt.ylabel("PC 2 loadings (33%)")
plt.scatter(pc1,pc2)
# label all the points on the scatterplot in accordance with the columns of 'table'
ax.annotate('k4fwd', xy=(pc1[0], pc2[0]))
ax.annotate('k5rev', xy=(pc1[1], pc2[1]))
ax.annotate('k6rev', xy=(pc1[2], pc2[2]))
ax.annotate('IL2Ra', xy=(pc1[3], pc2[3]))
ax.annotate('IL2Rb', xy=(pc1[4], pc2[4]))
ax.annotate('gc', xy=(pc1[5], pc2[5]))
ax.annotate('IL2_IL2Ra', xy=(pc1[6], pc2[6]))
ax.annotate('IL2_IL2Rb', xy=(pc1[7], pc2[7]))
ax.annotate('IL2_gc', xy=(pc1[8], pc2[8]))
ax.annotate('IL2_IL2Ra_IL2Rb', xy=(pc1[9], pc2[9]))
ax.annotate('IL2_IL2Ra_gc', xy=(pc1[10], pc2[10]))
ax.annotate('IL2_IL2Rb_gc', xy=(pc1[11], pc2[11]))
ax.annotate('IL2_IL2Ra_IL2Rb_gc', xy=(pc1[12], pc2[12]))
#show the plot that was just created
plt.show()

# generate loadings plot of PC1 vs PC3

# first need to do some set-up for my annotations
fig = plt.figure()
ax = fig.add_subplot(111)
# plot all the points from PC1 and PC3 in a scatter plot
plt.rcParams.update({'font.size': 8})
plt.xlabel("PC 1 loadings (33%)")
plt.ylabel("PC 3 loadings (33%)")
plt.scatter(pc1,pc3)
# label all the points on the scatterplot in accordance with the columns of 'table'
ax.annotate('k4fwd', xy=(pc1[0], pc3[0]))
ax.annotate('k5rev', xy=(pc1[1], pc3[1]))
ax.annotate('k6rev', xy=(pc1[2], pc3[2]))
ax.annotate('IL2Ra', xy=(pc1[3], pc3[3]))
ax.annotate('IL2Rb', xy=(pc1[4], pc3[4]))
ax.annotate('gc', xy=(pc1[5], pc3[5]))
ax.annotate('IL2_IL2Ra', xy=(pc1[6], pc3[6]))
ax.annotate('IL2_IL2Rb', xy=(pc1[7], pc3[7]))
ax.annotate('IL2_gc', xy=(pc1[8], pc3[8]))
ax.annotate('IL2_IL2Ra_IL2Rb', xy=(pc1[9], pc3[9]))
ax.annotate('IL2_IL2Ra_gc', xy=(pc1[10], pc3[10]))
ax.annotate('IL2_IL2Rb_gc', xy=(pc1[11], pc3[11]))
ax.annotate('IL2_IL2Ra_IL2Rb_gc', xy=(pc1[12], pc3[12]))
# show the plot that was just created
plt.show()


