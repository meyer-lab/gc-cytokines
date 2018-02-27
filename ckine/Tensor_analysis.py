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

factors = parafac(values,rank = 2)

#Generate a plot for component 1 vs component 2 of the factors[2] above representing our values
labels = ['Active IL2', 'Active IL15', 'Active IL7', 'Active IL9', 'Surface IL2Ra', 'Surface IL2Rb', 'Surface gc', 'Surface IL15Ra', 'Surface IL7Ra', 'Surface IL9R', 'Total IL2Ra', 'Total IL2Rb', 'Total gc', 'Total IL15Ra', 'Total IL7Ra', 'Total IL9R']
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(factors[2])):
    plt.scatter(factors[2][:,0][i], factors[2][:,1][i])
    ax.annotate(labels[i], xy=(factors[2][:,0][i], factors[2][:,1][i]), xytext = (0, 0), textcoords = 'offset points')

plt.xlabel('Component One')
plt.ylabel('Component Two')
plt.title('Values decomposition')
plt.show()


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

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(factors[0])):
    plt.scatter(factors[0][:,0][i], factors[0][:,1][i], color = 'k')
    #ax.annotate(str(i+1), xy=(factors[0][:,0][i], factors[0][:,1][i]), xytext = (0, 0), textcoords = 'offset points')
   

plt.xlabel('Component One')
plt.ylabel('Component Two')
plt.title('Combination Decomposition')
plt.show()
