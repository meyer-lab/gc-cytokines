"""
Generate a tensor for the different y-values that arise at different timepoints during the model and with various initial conditions. The initial conditions vary the concentrations of the ligands and the expression rates of the receptors to simulate different cell lines.
Important Notes:
    y_of_combos is a multidimensional matrix of size (length mesh x 1000 timeponts x 56 values of y)
    values is also a multidimensional matrix of size (length mesh x 1000 x 16 values for cytokine activity, surface receptors amount, and total receptors amount)
"""

import os
from os.path import join
import numpy as np, pandas as pds
from tqdm import trange
from .model import getTotalActiveCytokine, runCkineU, surfaceReceptors, totalReceptors, nParams, nSpecies, nRxn, internalStrength, halfL

path = os.path.dirname(os.path.abspath(__file__))
data = pds.read_csv(join(path, 'data/expr_table.csv')) # Every column in the data represents a specific cell

def ySolver(matIn, ts):
    """ This generates all the solutions of the tensor. """
    matIn = np.squeeze(matIn)

    # Set some given parameters already determined from fitting
    rxntfR = np.zeros(nParams())
    rxntfR[6] = 0.004475761 #kfwd
    rxntfR[7] = 8.543317686 #k4rev
    rxntfR[8] = 0.12321939  #k5rev
    rxntfR[9] = 3.107488811 #k16rev
    rxntfR[10] = 0.212958572 #k17rev
    rxntfR[11] = 0.013775029 #k22rev
    rxntfR[12] = 0.151523448 #k23rev
    rxntfR[13] = 0.094763588 #k27rev
    rxntfR[15] = 0.095618346 #k33rev
    rxntfR[[14, 16]] = 0.25  # From fitting IL9 and IL21: k4rev - k35rev
    rxntfR[17] = 0.080084184 #endo
    rxntfR[18] = 1.474695447 #activeEndo
    rxntfR[19] = 0.179927669 #sortF
    rxntfR[20] = 0.155260036 #kRec
    rxntfR[21] = 0.017236595 #kDeg

    rxntfR[22:30] = matIn[6:14] # Receptor expression
    rxntfR[0:6] = matIn[0:6] # Cytokine stimulation concentrations

    temp, retVal = runCkineU(ts, rxntfR)
    assert retVal >= 0

    return temp


def findy(lig, n_timepoints):
    """A function to find the different values of y at different timepoints and different initial conditions. Takes in how many ligand concentrations and expression rates to iterate over."""
    #Receptor expression levels were determined from the following cells through ImmGen
    #Expression Value Normalized by DESeq2, and we have 34 types of cells
    #Load the data from csv file
    path = os.path.dirname(os.path.abspath(__file__))
    data = pds.read_csv(join(path, 'data/expr_table.csv')) # Every column in the data represents a specific cell
    numpy_data = data.values # returns data values in a numpy array
    cell_names = data.columns.values.tolist()[1::] #returns the cell names from the pandas dataframe (which came from csv)

    #np.delete removes the first column of the data which only includes the name of the receptors (6x35 to 6x34)
    #['Il2ra' 'Il2rb' 'Il2rg' 'Il15ra' 'Il7r' 'Il9r', 'IL4Ra, 'IL21Ra'] in that order
    data_numbers = np.delete(numpy_data, 0, 1)

    ILs = np.logspace(-3., 2., num=lig) # Cytokine stimulation concentrations
    # Goal is to make one cell expression levels by len(mat) for every cell
    # Make mesh grid of all combinations of ligand
    mat = np.array(np.meshgrid(ILs, ILs, ILs, ILs, ILs, ILs)).T.reshape(-1, 6)

    mats = np.tile(mat, (len(cell_names), 1)) # Repeat the cytokine stimulations (mat) an X amount of times where X here is number of cells (34)
    receptor_repeats = np.repeat(data_numbers.T,len(mat), 0) #Create an array that repeats the receptor expression levels 'len(mat)' times
    new_mat = np.concatenate((mats, receptor_repeats), axis = 1) #concatenate to obtain the new meshgrid

    # generate n_timepoints evenly spaced timepoints to 4 hrs
    ts = np.logspace(-3., np.log10(4 * 60.), n_timepoints)
    ts = np.insert(ts, 0, 0.0)

    # Allocate a y_of_combos
    y_of_combos = np.zeros((len(new_mat), ts.size, nSpecies()))

    # Iterate through every combination of values and store solver values in a y matrix
    for ii in trange(new_mat.shape[0]):
        y_of_combos[ii] = ySolver(new_mat[ii,:], ts)

    return y_of_combos, new_mat, mat, mats, cell_names

def reduce_values(y_of_combos):
    """Reduce y_of_combinations into necessary values."""
    active_list = [np.array([7, 8, 14, 15]),np.array([18]),np.array([21]),np.array([24]),np.array([27])] #active indices for all receptors relative to cytokine; Note we combined the activity of IL2 and IL15
    values = np.zeros((y_of_combos.shape[0],y_of_combos.shape[1],21))
    indices = [np.array([0, 3, 5, 6, 8]), np.array([1, 4, 5, 7, 8, 11, 12, 14, 15]), np.array([2, 6, 7, 8, 13, 14, 15, 18, 21]), np.array([9, 10, 12, 13, 15]), np.array([16, 17, 18]), np.array([19, 20, 21]), np.array([22, 23, 24]),np.array([25, 26, 27])]
    for i in range(5): #first 6 total active cytokines
        values[:,:,i] = np.sum(y_of_combos[:,:,active_list[i]], axis = 2) + internalStrength() * np.sum(y_of_combos[:,:,halfL()+active_list[i]], axis = 2)
    for j in range(len(indices)):
        values[:,:,5+j] = np.sum(y_of_combos[:,:,indices[j]], axis = 2)
    for k in range(len(indices)):
        values[:,:,5+len(indices)+k] = values[:,:,5+k] + internalStrength() * np.sum(y_of_combos[:,:,halfL(): halfL() * 2][:,:,indices[k]], axis = 2)
    return values

def prepare_tensor(lig, n_timepoints = 100):
    """Function to generate the 4D values tensor."""
    y_of_combos, new_mat, mat, mats, cell_names = findy(lig, n_timepoints) #mat here is basically the 2^lig cytokine stimulation; mats

    values = reduce_values(y_of_combos)
    tensor4D = np.zeros((values.shape[1],len(cell_names),len(mat),values.shape[2]))
    for ii in range(tensor4D.shape[0]):
        tensor4D[ii] = values[:,ii,:].reshape(tensor4D.shape[1:4])
    return tensor4D, new_mat, mat, mats, cell_names
