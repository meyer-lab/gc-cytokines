"""
Generate a tensor for the different y-values that arise at different timepoints during the model and with various initial conditions.
The initial conditions vary the concentrations of the three ligands to simulate different cell lines.
Cell lines are defined by the number of each receptor subspecies on their surface.
"""
import os
from os.path import join
import numpy as np
import pandas as pds
from .model import runCkineU, nParams, nSpecies, runCkineU_IL2, getTotalActiveSpecies

# Set the following variables for multiple functions to use
endo = 0.08
kRec = 0.15
sortF = 0.18
kDeg = 0.017
kfwd = 0.004475761
k4rev = 8.543317686
k5rev = 0.12321939

def import_Rexpr():
    """ Loads CSV file containing Rexpr levels from Visterra data. """
    path = os.path.dirname(os.path.dirname(__file__))
    data = pds.read_csv(join(path, 'ckine/data/final_receptor_levels.csv'))  # Every row in the data represents a specific cell
    numpy_data = data.values[:, 1:]  # returns data values in a numpy array
    cell_names = list(data.values[:, 0])
    # ['Il2ra' 'Il2rb' 'Il2rg' 'Il15ra'] in that order from Receptor levels. CD25, CD122, CD132, CD215.
    return data, numpy_data, cell_names

def ySolver(matIn, ts):
    """ This generates all the solutions of the tensor. """
    matIn = np.squeeze(matIn)

    # Set some given parameters already determined from fitting
    rxntfR = np.zeros(nParams())
    rxntfR[6] = kfwd
    rxntfR[7] = k4rev
    rxntfR[8] = k5rev
    rxntfR[9] = 3.107488811  # k16rev
    rxntfR[10] = 0.212958572  # k17rev
    rxntfR[11] = 0.013775029  # k22rev
    rxntfR[12] = 0.151523448  # k23rev
    rxntfR[13] = 0.094763588  # k27rev
    rxntfR[15] = 0.095618346  # k33rev
    # TODO: Update parameters based on IL9&21.
    rxntfR[[14, 16]] = 0.15  # From fitting IL9 and IL21: k4rev - k35rev
    rxntfR[17] = endo  # endo
    rxntfR[18] = 1.46  # activeEndo
    rxntfR[19] = sortF  # sortF
    rxntfR[20] = kRec  # kRec
    rxntfR[21] = kDeg  # kDeg

    rxntfR[22:30] = matIn[6:14]  # Receptor expression

    rxntfR[0:6] = matIn[0:6]  # Cytokine stimulation concentrations in the following order: IL2, 15, 7, 9, 4, 21, and in nM

    temp, retVal = runCkineU(ts, rxntfR)
    assert retVal >= 0

    return temp


def ySolver_IL2(matIn, ts):
    """ This generates all the solutions of the tensor. """
    matIn = np.squeeze(matIn)

    k1rev = 0.6 * 10.0 * 0.01
    k2rev = 0.6 * 144.0
    k11rev = 63.0 * k5rev / 1.5
    rxntfr = np.array([matIn[0], kfwd, k1rev, k2rev, k4rev, k5rev, k11rev,
                       matIn[6], matIn[7], matIn[8],  # IL2Ra, IL2Rb, gc
                       k1rev * 5.0, k2rev * 5.0, k4rev * 5.0, k5rev * 5.0, k11rev * 5.0])

    yOut, retVal = runCkineU_IL2(ts, rxntfr)

    assert retVal >= 0

    return yOut


def findy(lig, n_timepoints):
    """A function to find the different values of y at different timepoints and different initial conditions. Takes in how many ligand concentrations and expression rates to iterate over."""
    # Load the data from csv file
    data, numpy_data, cell_names = import_Rexpr()
    ILs = np.logspace(-2., 1., num=lig)  # Cytokine stimulation concentrations in nM

    # Goal is to make one cell expression levels by len(mat) for every cell
    # Make mesh grid of all combinations of ligand
    mat = np.vstack((np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6),
                     np.array(np.meshgrid(0, ILs, 0, 0, 0, 0)).T.reshape(-1, 6),
                     np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6)))  # Set interleukins other than IL2&15 to zero. Should be of shape 3(IL2 IL15,mutIL2)*(num=lig) by 6 (6 for all ILs)

    mats = np.tile(mat, (len(cell_names), 1))  # Repeat the cytokine stimulations (mat) an X amount of times where X here is number of cells (8)

    # Set receptor levels for IL7Ra, IL9R, IL4Ra, IL21Ra to one. We won't use them for IL2-15 model. Second argument can also be set to 4 since we only have IL2Ra, IL2Rb, gc, IL15Ra measured.
    no_expression = np.ones((numpy_data.shape[0], 8 - numpy_data.shape[1])) * 0.0
    # need to convert numbers to expression values
    numpy_data[:, :] = (numpy_data[:, :] * endo) / (1. + ((kRec * (1. - sortF)) / (kDeg * sortF)))  # constant according to measured number per cell
    all_receptors = np.concatenate((numpy_data, no_expression), axis=1)  # Expression: IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL4Ra, IL21Ra in order
    receptor_repeats = np.repeat(all_receptors, len(mat), 0)  # Create an array that repeats the receptor expression levels 'len(mat)' times

    new_mat = np.concatenate((mats, receptor_repeats), axis=1)  # concatenate to obtain the new meshgrid

    # generate n_timepoints evenly spaced timepoints to 4 hrs
    ts = np.logspace(-3., np.log10(4 * 60.), n_timepoints)
    ts = np.insert(ts, 0, 0.0)

    # Allocate a y_of_combos
    y_of_combos = np.zeros((len(new_mat), ts.size, nSpecies()))

    # Iterate through every combination of values and store solver values in a y matrix
    for ii in range(new_mat.shape[0]):
        if ii % len(mat) > 7:
            y_of_combos[ii] = ySolver_IL2(new_mat[ii, :], ts)
        else:
            y_of_combos[ii] = ySolver(new_mat[ii, :], ts)

    return y_of_combos, new_mat, mat, mats, cell_names


def prepare_tensor(lig, n_timepoints=100):
    """Function to generate the 4D values tensor."""
    y_of_combos, new_mat, mat, mats, cell_names = findy(lig, n_timepoints)  # mat here is basically the 2^lig cytokine stimulation; mats

    values = np.zeros((y_of_combos.shape[0], y_of_combos.shape[1], 1))

    values[:, :, 0] = np.tensordot(y_of_combos, getTotalActiveSpecies(), (2, 0))

    tensor3D = np.zeros((values.shape[1], len(cell_names), len(mat)))

    for ii in range(tensor3D.shape[0]):
        tensor3D[ii] = values[:, ii, 0].reshape(tensor3D.shape[1:3])

    return tensor3D, new_mat, mat, mats, cell_names
