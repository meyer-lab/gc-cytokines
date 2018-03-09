"""
Analyze tensor from Sampling.pickle and plotting.
"""
import os
import pickle
import numpy as np
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data/Tensor_results/Sampling.pickle")
with open(filename, 'rb') as file:
    sample = pickle.load(file)

mat, values = sample[0], sample[1]


def z_score_values(A):
    '''Function that takes in the values tensor and z-scores it.'''
    B = np.zeros_like(A)
    for i in range(A.shape[2]):
        slice = A[:,:,i]
        mu = np.mean(slice)
        sigma = np.std(slice)
        z_scored_slice = (slice - mu)/sigma
        B[:,:,i] = z_scored_slice
    return B

values_z = z_score_values(values)

#Find which combinations have low gc expression vs high gc expression
gc_low = np.zeros(len(mat))
IL2Ra_low = np.zeros(len(mat))
IL2Rb_low = np.zeros(len(mat))
IL15Ra_low = np.zeros(len(mat))
IL7Ra_low = np.zeros(len(mat))
IL9R_low = np.zeros(len(mat))
IL2_low = np.zeros(len(mat))
IL15_low = np.zeros(len(mat))
IL7_low = np.zeros(len(mat))
IL9_low = np.zeros(len(mat))

gc_high = np.zeros(len(mat))
IL2Ra_high = np.zeros(len(mat))
IL2Rb_high = np.zeros(len(mat))
IL15Ra_high = np.zeros(len(mat))
IL7Ra_high = np.zeros(len(mat))
IL9R_high = np.zeros(len(mat))
IL2_high = np.zeros(len(mat))
IL15_high = np.zeros(len(mat))
IL7_high = np.zeros(len(mat))
IL9_high = np.zeros(len(mat))

#Find where receptors are low and high

l1,l2,l3,l4,l5,l6,l7,l8,l9,l10 = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

a1, a2, a3, a4 = -1,-1,-1,-1
b1, b2, b3, b4 = -1, -1, -1, -1

for k in range(len(mat)):
    #Focus of ligand concentrations: high vs low
    for i in range(4):
        if mat[k,i] == 1e-3: #Condition for low ligand concentration
            if i == 0:
                a1 = a1 + 1
                IL2_low[a1] = k
            elif i == 1:
                a2 = a2 + 1
                IL15_low[a2] = k
            elif i == 2:
                a3 = a3 + 1
                IL7_low[a3] = k
            elif i== 3:
                a4 = a4 +1
                IL9_low[a4] = k
        elif mat[k,i] == 1e3: #Condition for high ligand concentration
            if i == 0:
                b1 = b1 + 1
                IL2_high[b1] = k
            elif i == 1:
                b2 = b2 + 1
                IL15_high[b2] = k
            elif i == 2:
                b3 = b3 + 1
                IL7_high[b3] = k
            elif i== 3:
                b4 = b4 +1
                IL9_high[b4] = k
    #Focus on receptor expression rates: high vs low
    for i in range(4,10):
        if mat[k,i] == 1e-3: #Condition for high receptor expression rate
            if i == 4:
                l4 = l4+1
                IL2Ra_low[l4] = k
            elif i == 5:
                l5 = l5+1
                IL2Rb_low[l5] = k
            elif i == 6:
                l6 = l6 + 1
                gc_low[l6] = k
            elif i == 7:
                l7 = l7+1
                IL15Ra_low[l7] = k
            elif i == 8:
                l8 = l8 + 1
                IL7Ra_low[l8] = k
            else:
                l9 = l9 + 1
                IL9R_low[l9] = k
        else: #Condition for low receptor expression rate
            if i == 4:
                h4 = h4+1
                IL2Ra_high[h4] = k
            elif i == 5:
                h5 = h5+1
                IL2Rb_high[h5] = k
            elif i == 6:
                h6 = h6 + 1
                gc_high[h6] = k
            elif i == 7:
                h7 = h7+1
                IL15Ra_high[h7] = k
            elif i == 8:
                h8 = h8 + 1
                IL7Ra_high[h8] = k
            else:
                h9 = h9 + 1
                IL9R_high[h9] = k

#Remove trailing Zeros from the preallocated arrays to avoid appending

IL2Ra_low = np.trim_zeros(IL2Ra_low, 'b')
IL2Rb_low = np.trim_zeros(IL2Rb_low, 'b')
gc_low = np.trim_zeros(gc_low, 'b')
IL15Ra_low = np.trim_zeros(IL15Ra_low, 'b')
IL7Ra_low = np.trim_zeros(IL7Ra_low, 'b')
IL9R_low = np.trim_zeros(IL9R_low, 'b')
IL2_low = np.trim_zeros(IL2_low, 'b')
IL15_low = np.trim_zeros(IL15_low, 'b')
IL7_low = np.trim_zeros(IL7_low, 'b')
IL9_low = np.trim_zeros(IL9_low, 'b')
#put all the above in a tensor to allow for iterative plotting
Values_low_list = [IL2Ra_low, IL2Rb_low, gc_low, IL15Ra_low, IL7Ra_low, IL9R_low, IL2_low, IL15_low, IL7_low, IL9_low]

IL2Ra_high = np.trim_zeros(IL2Ra_high, 'b')
IL2Rb_high = np.trim_zeros(IL2Rb_high, 'b')
gc_high = np.trim_zeros(gc_high, 'b')
IL15Ra_high = np.trim_zeros(IL15Ra_high, 'b')
IL7Ra_high = np.trim_zeros(IL7Ra_high, 'b')
IL9R_high = np.trim_zeros(IL9R_high, 'b')
IL2_high = np.trim_zeros(IL2_high, 'b')
IL15_high = np.trim_zeros(IL15_high, 'b')
IL7_high = np.trim_zeros(IL7_high, 'b')
IL9_high = np.trim_zeros(IL9_high, 'b')
Values_high_list = [IL2Ra_high, IL2Rb_high, gc_high, IL15Ra_high, IL7Ra_high, IL9R_high, IL2_high, IL15_high, IL7_high, IL9_high]


#Perform Parafac tensor decomposition
factors = parafac(values_z,rank = 2)

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

#Generate a plot for component 1 vs component 2 of the factors[1] above representing our timepoints
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

#Generate a plot for component 1 vs component 2 of the factors[0] above representing our combinations
for i in range(10):
    fig = plt.figure()
    count = 0
    for j in Values_low_list[i]:
        j = int(j)
        plt.scatter(factors[0][:,0][j], factors[0][:,1][j], color = 'r', alpha = 0.2)
        count = count+1
    for j in Values_high_list[i]:
        j = int(j)
        plt.scatter(factors[0][:,0][j], factors[0][:,1][j], color = 'b', alpha = 0.2)
    plt.xlabel('Component One')
    plt.ylabel('Component Two')
    #Choose which title to add
    if i==0:
        #title the IL2Ra
        plt.title('Combination Decomposition: Low IL2Ra (red) vs High IL2Ra (blue)')
    elif i == 1:
        #title the IL2Rb
        plt.title('Combination Decomposition: Low IL2Rb (red) vs High IL2Rb (blue)')
    elif i == 2:
        plt.title('Combination Decomposition: Low gc (red) vs High gc (blue)')
    elif i ==3:
        plt.title('Combination Decomposition: Low IL15Ra (red) vs High IL15Ra (blue)')
    elif i == 4:
        plt.title('Combination Decomposition: Low IL7Ra (red) vs High IL7Ra (blue)')
    elif i == 5:
        plt.title('Combination Decomposition: Low IL9R (red) vs High IL9R (blue)')
    elif i == 6:
        plt.title('Combination Decomposition: Low IL2 (red) vs High IL2 (blue)')
    elif i == 7:
        plt.title('Combination Decomposition: Low IL15 (red) vs High IL15 (blue)')
    elif i == 8:
        plt.title('Combination Decomposition: Low IL7 (red) vs High IL7 (blue)')
    elif i == 9:
        plt.title('Combination Decomposition: Low IL9 (red) vs High IL9 (blue)')
    plt.show()
