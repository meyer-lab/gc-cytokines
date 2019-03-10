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
from .model import runCkineU, nParams, nSpecies, internalStrength, halfL

#Load the data from csv file
path = os.path.dirname(os.path.abspath(__file__))
data = pds.read_csv(join(path, 'data/Preliminary receptor levels.csv')) # Every row in the data represents a specific cell
numpy_data = data.values[:,1:] # returns data values in a numpy array
cell_names = list(data.values[:,0]) #returns the cell names from the pandas dataframe (which came from csv). 8 cells. 
#['Il2ra' 'Il2rb' 'Il2rg' 'Il15ra'] in that order from Receptor levels. CD25, CD122, CD132, CD215

#Set the following variables for multiple functions to use
endo = 0.080084184
kRec = 0.155260036
sortF = 0.179927669
kDeg = 0.017236595

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
    #TODO: Update parameters based on IL9&21.
    rxntfR[[14, 16]] = 0.15  # From fitting IL9 and IL21: k4rev - k35rev
    rxntfR[17] = endo #endo
    rxntfR[18] = 1.474695447 #activeEndo
    rxntfR[19] = sortF #sortF
    rxntfR[20] = kRec #kRec
    rxntfR[21] = kDeg #kDeg

    rxntfR[22:30] = matIn[6:14] # Receptor expression
    
    rxntfR[0:6] = matIn[0:6] # Cytokine stimulation concentrations in the following order: IL2, 15, 7, 9, 4, 21, and in nM

    temp, retVal = runCkineU(ts, rxntfR)
    assert retVal >= 0

    return temp


def findy(lig, n_timepoints):
    """A function to find the different values of y at different timepoints and different initial conditions. Takes in how many ligand concentrations and expression rates to iterate over."""

    ILs = np.logspace(-3., 2., num=lig) # Cytokine stimulation concentrations
    # Goal is to make one cell expression levels by len(mat) for every cell
    # Make mesh grid of all combinations of ligand
    mat = np.array(np.meshgrid(ILs, ILs, 0, 0, 0, 0)).T.reshape(-1, 6) #Set interleukins other than IL2&15 to zero
    mats = np.tile(mat, (len(cell_names), 1)) # Repeat the cytokine stimulations (mat) an X amount of times where X here is number of cells (8)

    no_expression = np.ones((numpy_data.shape[0], 8 - numpy_data.shape[1])) * 0.5 #Set receptor levels for IL7Ra, IL9R, IL4Ra, IL21Ra to one. We won't use them for IL2-15 model. Second argument can also be set to 4 since we only have IL2Ra, IL2Rb, gc, IL15Ra measured. 
    #need to convert numbers to expression values
    for ii in range(numpy_data.shape[1]):
        numpy_data[:,ii] = (numpy_data[:,ii] * endo) / (1. + ((kRec*(1.-sortF)) / (kDeg*sortF))) # constant according to measured number per cell
    all_receptors = np.concatenate((numpy_data, no_expression), axis = 1) #Expression: IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL4Ra, IL21Ra in order
    receptor_repeats = np.repeat(all_receptors,len(mat), 0) #Create an array that repeats the receptor expression levels 'len(mat)' times

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
    values = np.zeros((y_of_combos.shape[0],y_of_combos.shape[1],5)) #Select 5 for IL2+15, IL7, IL9, IL4, IL21
    indices = [np.array([0, 3, 5, 6, 8]), np.array([1, 4, 5, 7, 8, 11, 12, 14, 15]), np.array([2, 6, 7, 8, 13, 14, 15, 18, 21]), np.array([9, 10, 12, 13, 15]), np.array([16, 17, 18]), np.array([19, 20, 21]), np.array([22, 23, 24]),np.array([25, 26, 27])]
    for i in range(5): #first 6 total active cytokines
        values[:,:,i] = np.sum(y_of_combos[:,:,active_list[i]], axis = 2) + internalStrength()*np.sum(y_of_combos[:,:,halfL()+active_list[i]], axis = 2)
    return values

def prepare_tensor(lig, n_timepoints = 100):
    """Function to generate the 4D values tensor."""
    y_of_combos, new_mat, mat, mats, cell_names = findy(lig, n_timepoints) #mat here is basically the 2^lig cytokine stimulation; mats

    values = reduce_values(y_of_combos)
    tensor3D = np.zeros((values.shape[1],len(cell_names),len(mat)))
    for ii in range(tensor3D.shape[0]):
        tensor3D[ii] = values[:,ii,0].reshape(tensor3D.shape[1:3])
    return tensor3D, new_mat, mat, mats, cell_names
