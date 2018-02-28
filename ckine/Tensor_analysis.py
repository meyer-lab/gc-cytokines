"""
Analyze tensor from Sampling.pickle
"""
import os
import pickle
import tensorly as tl
import numpy as np
from tensorly.decomposition import partial_tucker, tucker, parafac
import matplotlib.pyplot as plt


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data/Tensor_results/Sampling.pickle")
with open(filename, 'rb') as file:
    sample = pickle.load(file)

mat, values = sample[0], sample[1]

#Find which combinations have low gc expression vs high gc expression
gc_low = np.zeros(len(mat))
gc_high = np.zeros(len(mat))
l = -1
h = -1
for k in range(len(mat)):
    if mat[k,6] == 1e-3:
        l = l+1
        gc_low[l] = k
    else:
        h = h+1
        gc_high[h] = k
#Remove trailing Zeros
gc_low = np.trim_zeros(gc_low, 'b')
gc_high = np.trim_zeros(gc_high, 'b')

#Perform Parafac tensor decomposition
factors = parafac(values,rank = 2)

#Generate a plot for component 1 vs component 2 of the factors[2] above representing our values
labels = ['IL2', 'IL15', 'IL7', 'IL9', 'IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R']
fig = plt.figure()
ax = fig.add_subplot(111)

#Set Active to color red
#Set Surface to color blue
# Set Total to color black
for i in range(len(factors[2])):
    if i in range(4):
        c = 'r'
        if i==0:
            plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c, label = 'Ligand Activity')
        else:
            plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c)
        ax.annotate(labels[i], xy=(factors[2][:,0][i], factors[2][:,1][i]), xytext = (0, 0), textcoords = 'offset points')
    elif i in range(4,10):
        c = 'b'
        if i == 4:
            plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c, label = 'Surface Receptor')
        else:
            plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c)
        ax.annotate(labels[i], xy=(factors[2][:,0][i], factors[2][:,1][i]), xytext = (0, 0), textcoords = 'offset points')
    else:
        c = 'k'
        if i==10:
            plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c, label = 'Total Receptor')
        else:
            plt.scatter(factors[2][:,0][i], factors[2][:,1][i], color = c)
        ax.annotate(labels[i], xy=(factors[2][:,0][i], factors[2][:,1][i]), xytext = (0, 0), textcoords = 'offset points')

plt.xlabel('Component One')
plt.ylabel('Component Two')
plt.title('Values decomposition')
plt.legend()
plt.show()

#Generate a plot for component 1 vs component 2 of the factors[1] above representing our values
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(factors[1])):
    plt.scatter(factors[1][:,0][i], factors[1][:,1][i], color = 'k')
    #ax.annotate(str(i+1), xy=(factors[1][:,0][i], factors[1][:,1][i]), xytext = (0, 0), textcoords = 'offset points')
    if i == 99:
        ax.annotate(str(i+1), xy=(factors[1][:,0][i], factors[1][:,1][i]), xytext = (0, 0), textcoords = 'offset points')

plt.xlabel('Component One')
plt.ylabel('Component Two')
plt.title('Timepoint Decomposition')
plt.show()

#Generate a plot for component 1 vs component 2 of the factors[0] above representing our values
fig = plt.figure()
ax = fig.add_subplot(111)
#for i in range(len(factors[0])):
    
    #plt.scatter(factors[0][:,0][i], factors[0][:,1][i], color = 'k')
    #ax.annotate(str(i+1), xy=(factors[0][:,0][i], factors[0][:,1][i]), xytext = (0, 0), textcoords = 'offset points')

#Color low gc red, high gc blue
for i in gc_low:
    i = int(i)
    plt.scatter(factors[0][:,0][i], factors[0][:,1][i], color = 'r')
for i in gc_high:
    i = int(i)
    plt.scatter(factors[0][:,0][i], factors[0][:,1][i], color = 'b')

plt.xlabel('Component One')
plt.ylabel('Component Two')
plt.title('Combination Decomposition: Low gc (red) vs High gc (blue)')
plt.show()
