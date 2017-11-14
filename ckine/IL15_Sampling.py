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
ys = np.zeros((729,10))
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

# generate scores plot of PC1 vs PC2
plt.rcParams.update({'font.size': 8})
plt.xlabel("PC 1 scores (29.57%)")
plt.ylabel("PC 2 scores (18.98%)")
plt.scatter(scores[:,0], scores[:,1])
plt.show()

#generate scores plot of PC1 vs PC3
plt.rcParams.update({'font.size': 8})
plt.xlabel("PC 1 scores (29.57%)")
plt.ylabel("PC 3 scores (11.24%)")
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
plt.xlabel("PC 1 loadings (29.57%)")
plt.ylabel("PC 2 loadings (11.24%)")
plt.scatter(pc1,pc2)

# label all the points on the scatterplot in accordance with the columns of 'table'
ax.annotate('k13fwd', xy=(pc1[0], pc2[0]))
ax.annotate('k15rev', xy=(pc1[1], pc2[1]))
ax.annotate('k17rev', xy=(pc1[2], pc2[2]))
ax.annotate('k18rev', xy=(pc1[3], pc2[3]))
ax.annotate('k22rev', xy=(pc1[4], pc2[4]))
ax.annotate('k23rev', xy=(pc1[5], pc2[5]))
ax.annotate('IL2Rb', xy=(pc1[6], pc2[6]))
ax.annotate('gc', xy=(pc1[7], pc2[7]))
ax.annotate('IL15Ra', xy=(pc1[8], pc2[8]))
ax.annotate('IL15_IL15Ra', xy=(pc1[9], pc2[9]))
ax.annotate('IL15_IL2Rb', xy=(pc1[10], pc2[10]))
ax.annotate('IL15_gc', xy=(pc1[11], pc2[11]))
ax.annotate('IL15_IL15Ra_IL2Rb', xy=(pc1[12], pc2[12]))
ax.annotate('IL15_IL15Ra_gc', xy=(pc1[13], pc2[13]))
ax.annotate('IL15_IL2Rb_gc', xy=(pc1[14], pc2[14]))
ax.annotate('IL15_IL15Ra_IL2Rb_gc', xy=(pc1[15], pc2[15]))
#show the plot that was just created
plt.show()

# generate loadings plot of PC1 vs PC3

# first need to do some set-up for my annotations
fig = plt.figure()
ax = fig.add_subplot(111)
# plot all the points from PC1 and PC3 in a scatter plot
plt.rcParams.update({'font.size': 8})
plt.xlabel("PC 1 loadings (29.57%)")
plt.ylabel("PC 3 loadings (11.24%)")
plt.scatter(pc1,pc3)
# label all the points on the scatterplot in accordance with the columns of 'table'
ax.annotate('k13fwd', xy=(pc1[0], pc3[0]))
ax.annotate('k15rev', xy=(pc1[1], pc3[1]))
ax.annotate('k17rev', xy=(pc1[2], pc3[2]))
ax.annotate('k18rev', xy=(pc1[3], pc3[3]))
ax.annotate('k22rev', xy=(pc1[4], pc3[4]))
ax.annotate('k23rev', xy=(pc1[5], pc3[5]))
ax.annotate('IL2Rb', xy=(pc1[6], pc3[6]))
ax.annotate('gc', xy=(pc1[7], pc3[7]))
ax.annotate('IL15Ra', xy=(pc1[8], pc3[8]))
ax.annotate('IL15_IL15Ra', xy=(pc1[9], pc3[9]))
ax.annotate('IL15_IL2Rb', xy=(pc1[10], pc3[10]))
ax.annotate('IL15_gc', xy=(pc1[11], pc3[11]))
ax.annotate('IL15_IL15Ra_IL2Rb', xy=(pc1[12], pc3[12]))
ax.annotate('IL15_IL15Ra_gc', xy=(pc1[13], pc3[13]))
ax.annotate('IL15_IL2Rb_gc', xy=(pc1[14], pc3[14]))
ax.annotate('IL15_IL15Ra_IL2Rb_gc', xy=(pc1[15], pc3[15]))
#show the plot that was just created
plt.show()
