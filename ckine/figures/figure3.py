"""
This creates Figure 3.
"""
import os
from os.path import join
import pickle
import string
import time
import tensorly as tl
import numpy as np, pandas as pds
from scipy import stats
from sklearn.decomposition.pca import PCA
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, plot_cells, plot_ligands, plot_timepoints
from ..Tensor_analysis import find_R2X, scale_all, perform_decomposition, perform_tucker, find_R2X_tucker
from ..tensor_generation import data, prepare_tensor

n_ligands = 4
values, _, mat, _, _ = prepare_tensor(n_ligands)
values = tl.tensor(values)

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    x, y = 3, 4
    ax, f = getSetup((16, 14), (x, y))
    # Blank out for the cartoon
    ax[0].axis('off')
    ax[8].axis('off')

    factors_activity = []
    for jj in range(len(mat) - 1):
        tic = time.clock()
        print(jj)
        factors = perform_decomposition(values , jj+1)
        factors_activity.append(factors)
    toc = time.clock()
    print(toc - tic)


    numpy_data = data.values[:,1:] # returns data values in a numpy array
    cell_names = list(data.values[:,0]) #returns the cell names from the pandas dataframe (which came from csv). 8 cells. 
    #['Il2ra' 'Il2rb' 'Il2rg' 'Il15ra'] in that order from Receptor levels. CD25, CD122, CD132, CD215

    n_comps = 4
    factors_activ = factors_activity[n_comps-1]
    newfactors_activ = factors_activ
    newfactors = scale_all(newfactors_activ)

    PCA_receptor(ax[1], ax[2], cell_names, numpy_data.T)
    plot_R2X(ax[3], values, factors_activity, n_comps = 14)

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    plot_timepoints(ax[4], newfactors[0]) #Change final input value depending on need

    for row in range(1,3):
        subplotLabel(ax[row], string.ascii_uppercase[row]) # Add subplot labels
        compNum = 2*(row-1) + 1
        plot_cells(ax[row*y + 1], newfactors[1], compNum, compNum+1, cell_names, ax_pos = row*y + 1, legend=True)
        plot_ligands(ax[row*y + 2], newfactors[2], compNum, compNum+1)

        # Set axes to center on the origin, and add labels
        for col in range(1,y):
            ax[row*y + col].set_xlabel('Component ' + str(compNum))
            ax[row*y + col].set_ylabel('Component ' + str(compNum+1))

            x_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_xlim())))*1.1
            y_max = np.max(np.absolute(np.asarray(ax[row*y + col].get_ylim())))*1.1

            ax[row*y + col].set_xlim(-x_max, x_max)
            ax[row*y + col].set_ylim(-y_max, y_max)

    #f.tight_layout()

    return f

def PCA_receptor(ax1, ax2, cell_names, data):
    """Plot PCA scores and loadings for Receptor Expression Data. """
    pca = PCA(n_components = 2)
    data = stats.zscore(data.astype(float), axis = 0)
    scores = pca.fit(data.T).transform(data.T) #34 cells by n_comp
    loadings = pca.components_ #n_comp by 8 receptors
    expVar = pca.explained_variance_ratio_

    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '4', 'H']
    markersReceptors = ['^', '4', 'P', '*', 'D', 's', 'X' ,'o']
    labelReceptors = ['IL-2Rα', 'IL-2Rβ', r'$\gamma_{c}$', 'IL-15Rα', 'IL-7Rα', 'IL-9R', 'IL-4Rα', 'IL-21Rα']

    for ii in range(scores.shape[0]):
        ax1.scatter(scores[ii,0], scores[ii,1], c = colors[ii], marker = markersCells[ii], label = cell_names[ii])

    for jj in range(loadings.shape[1]):
        ax2.scatter(loadings[0,jj], loadings[1,jj], marker = markersReceptors[jj], label = labelReceptors[jj])

    x_max1 = np.max(np.absolute(np.asarray(ax1.get_xlim())))*1.1
    y_max1 = np.max(np.absolute(np.asarray(ax1.get_ylim())))*1.1

    x_max2 = np.max(np.absolute(np.asarray(ax2.get_xlim())))*1.1
    y_max2 = np.max(np.absolute(np.asarray(ax2.get_ylim())))*1.1

    ax1.set_xlim(-x_max1, x_max1)
    ax1.set_ylim(-y_max1, y_max1)
    ax1.set_xlabel('PC1 (' + str(round(expVar[0]*100, 2))+ '%)')
    ax1.set_ylabel('PC2 (' + str(round(expVar[1]*100, 2))+ '%)')
    ax1.set_title('Scores')
    ax1.legend(loc='upper left', bbox_to_anchor=(3.5, 1.0))

    ax2.set_xlim(-x_max2, x_max2)
    ax2.set_ylim(-y_max2, y_max2)
    ax2.set_xlabel('PC1 (' + str(round(expVar[0]*100, 2))+ '%)')
    ax2.set_ylabel('PC2 (' + str(round(expVar[1]*100, 2))+ '%)')
    ax2.set_title('Loadings')
    ax2.legend()

def plot_R2X(ax, tensor, factors_list, n_comps):
    """Function to plot R2X bar graph."""
    R2X_array = list()
    for n in range(n_comps):
        factors = factors_list[n]
        R2X = find_R2X(tensor, factors)
        R2X_array.append(R2X)
    ax.plot(range(1,n_comps+1), R2X_array, 'ko', label = 'Overall R2X')
    ax.set_ylabel('R2X')
    ax.set_xlabel('Number of Components')
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, n_comps+1))
    ax.set_xticklabels(np.arange(1, n_comps+1))
