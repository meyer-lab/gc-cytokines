"""
Generate a tensor for the different y-values that arise at different timepoints during the model and with various initial conditions. The initial conditions vary the concentrations of the ligands and the expression rates of the receptors to simulate different cell lines.
Important Notes:
    y_of_combos is a multidimensional matrix of size (length mesh x 1000 timeponts x 56 values of y)
    values is also a multidimensional matrix of size (length mesh x 1000 x 16 values for cytokine activity, surface receptors amount, and total receptors amount)
"""

import os
from os.path import join
import numpy as np, pandas as pds
from tqdm import tqdm
from multiprocessing import Pool
from .model import getTotalActiveCytokine, runCkineU, surfaceReceptors, totalReceptors, nParams, nSpecies

path = os.path.dirname(os.path.abspath(__file__))
data = pds.read_csv(join(path, 'data/expr_table.csv')) # Every column in the data represents a specific cell

ts = np.linspace(0.0, 4 * 60., 1000) # generate 1000 evenly spaced timepoints to 4 hrs


def ySolver(matIn):
    """ This generates all the solutions of the tensor. """
    matIn = np.squeeze(matIn)

    # Set some given parameters already determined from fitting
    rxntfR = np.zeros(nParams())
    rxntfR[6] = 0.00001 #kfwd
    rxntfR[7:17] = 0.001  # From fitting: k4rev - k35rev
    rxntfR[17:22] = 0.1 # From fitting: endo - kdeg

    rxntfR[22:30] = matIn[6:14] # Receptor expression
    rxntfR[0:6] = matIn[0:6] # Cytokine stimulation concentrations

    temp, retVal = runCkineU(ts, rxntfR)
    assert retVal >= 0

    return temp


def findy(lig):
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

    ILs = np.logspace(-3, 2, num=lig) # Cytokine stimulation concentrations
    # Goal is to make one cell expression levels by len(mat) for every cell
    # Make mesh grid of all combinations of ligand
    mat = np.array(np.meshgrid(ILs, ILs, ILs, ILs,ILs,ILs)).T.reshape(-1, 6)

    mats = np.tile(mat, (len(cell_names), 1)) # Repeat the cytokine stimulations (mat) an X amount of times where X here is number of cells (34)
    receptor_repeats = np.repeat(data_numbers.T,len(mat), 0) #Create an array that repeats the receptor expression levels 'len(mat)' times
    new_mat = np.concatenate((mats, receptor_repeats), axis = 1) #concatenate to obtain the new meshgrid

    # Allocate a y_of_combos
    y_of_combos = np.zeros((len(new_mat), ts.size, nSpecies()))

    pool = Pool()

    # Iterate through every combination of values and store solver values in a y matrix
    ii = 0
    for x in tqdm(pool.imap(ySolver, np.split(new_mat, new_mat.shape[0])), total=new_mat.shape[0]):
        y_of_combos[ii] = x
        ii = ii + 1

    pool.close()
    pool.join()

    return y_of_combos, new_mat, mat, mats, cell_names


def activity_surface_total(yVec):
    """This function returns a vector of 16 elements where the activity of the 6 cytokines and amounts of surface and total receptors are included."""
    x = np.zeros(22)

    for ii in range(6):
        x[ii] = getTotalActiveCytokine(ii, yVec)

    x[6:14] = surfaceReceptors(yVec)
    x[14:22] = totalReceptors(yVec)
    return x


def activity_surf_tot(y_of_combos):
    """This function returns the activity and amounts of receptors both on the surface and total for every timepoint per combination of values"""
    values = np.zeros((y_of_combos.shape[0], y_of_combos.shape[1], 22))

    for i in range(y_of_combos.shape[0]):
        for j in range(y_of_combos.shape[1]):
            values[i][j] = activity_surface_total(y_of_combos[i][j])

    return values
