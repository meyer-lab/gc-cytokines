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
from .model import getTotalActiveCytokine, runCkineU, surfaceReceptors, totalReceptors

path = os.path.dirname(os.path.abspath(__file__))
data = pds.read_csv(join(path, 'data/expr_table.csv')) # Every column in the data represents a specific cell

def findy(lig, timelength = 1000):
    """A function to find the different values of y at different timepoints and different initial conditions. Takes in how many ligand concentrations and expression rates to iterate over."""
    #Receptor expression levels were determined from the following cells through ImmGen
    #Expression Value Normalized by DESeq2, and we have 34 types of cells
    #Load the data from csv file
    path = os.path.dirname(os.path.abspath(__file__))
    data = pds.read_csv(join(path, 'data/expr_table.csv')) # Every column in the data represents a specific cell
    numpy_data = data.values # returns data values in a numpy array
    cell_names = data.columns.values.tolist()[1::] #returns the cell names from the pandas dataframe (which came from csv)

    #np.delete removes the first column of the data which only includes the name of the receptors (6x35 to 6x34)
    #['Il2ra' 'Il2rb' 'Il2rg' 'Il15ra' 'Il7r' 'Il9r'] in that order
    data_numbers = np.delete(numpy_data,0,1)

    t = 60. * 4 # let's let the system run for 4 hours
    ts = np.linspace(0.0, t, timelength) #generate 1000 evenly spaced timepoints
    IL2 = IL15 = IL7 = IL9 = np.logspace(-3, 3, num=lig) #cytokine stimulation concentrations
    #Goal is to make one cell expresison levels by len(mat) for every cell
    mat = np.array(np.meshgrid(IL2,IL15,IL7,IL9)).T.reshape(-1,4)
    mats = np.tile(mat,(len(cell_names),1)) # Repeat the cytokine stimulations (mat) an X amount of times where X here is number of cells (34)
    receptor_repeats = np.repeat(data_numbers.T,len(mat), 0) #Create an array that repeats the receptor expression levels 'len(mat)' times
    new_mat = np.concatenate((mats,receptor_repeats), axis = 1) #concatenate to obtain the new meshgrid

    #Preset a y_of_combos
    y_of_combos = np.zeros((len(new_mat), len(ts),48))

    #Set some given parameters already determined from fitting
    rxntfR = np.zeros(24)
    rxntfR[4:13] = np.ones(9) * (5*10**-1)  #From fitting: kfwd - k31rev
    rxntfR[13:18] = np.ones(5) * (50* 10**-3) #From fitting: endo - kdeg

    #Iterate through every combination of values and store solver values in a y matrix
    for ii in tqdm(range(len(new_mat))):
        #Create a new y0 everytime odeint is run per combination of values.
        rxntfR[18:24] = new_mat[ii,4:10]
        rxntfR[0:4] = new_mat[ii,0:4] #Cytokine stimulation concentrations

        temp, retVal = runCkineU(ts, rxntfR)
        assert retVal >= 0
        y_of_combos[ii] = temp # only assign values to ys if there isn't an error message; all errors will still be 0
    return y_of_combos, new_mat, mat, mats, cell_names

def activity_surface_total(yVec):
    """This function returns a vector of 16 elements where the activity of the 4 cytokines and amounts of surface and total receptors are included."""
    x = np.zeros(16)
    x[0],x[1],x[2],x[3] = getTotalActiveCytokine(0,yVec), getTotalActiveCytokine(1,yVec), getTotalActiveCytokine(2,yVec), getTotalActiveCytokine(3,yVec)
    x[4:10] = surfaceReceptors(yVec)
    x[10:16] = totalReceptors(yVec)
    return x

def activity_surf_tot(y_of_combos):
    """This function returns the activity and amounts of receptors both on the surface and total for every timepoint per combination of values"""
    values = np.zeros((len(y_of_combos), y_of_combos.shape[1],16))

    for i, _ in enumerate(y_of_combos):
        for j in range(y_of_combos.shape[1]):
            values[i][j] = activity_surface_total(y_of_combos[i][j])
    return values
