"""
Generate a tensor for the different y-values that arise at different timepoints during the model and with various initial conditions.
The initial conditions vary the concentrations of the three ligands to simulate different cell lines.
Cell lines are defined by the number of each receptor subspecies on their surface.
"""
import numpy as np
from .model import runCkineU, nSpecies, runCkineU_IL2, getTotalActiveSpecies
from .imports import import_Rexpr, import_samples_2_15, import_pstat

rxntfR, _ = import_samples_2_15(N=1, tensor=True)
rxntfR = np.squeeze(rxntfR)

n_lig = 3  # Set the number of different cytokines used to make the tensor to 3


def ySolver(matIn, ts, tensor=True):
    """ This generates all the solutions for the Wild Type interleukins across conditions defined in meshprep(). """
    matIn = np.squeeze(matIn)
    if tensor:
        rxntfR[22:30] = matIn[6:14]  # Receptor expression
    rxntfR[0:6] = matIn[0:6]  # Cytokine stimulation concentrations in the following order: IL2, 15, 7, 9, 4, 21, and in nM

    temp, retVal = runCkineU(ts, rxntfR)
    assert retVal >= 0

    return temp


def ySolver_IL2_mut(matIn, ts):
    """ This generates all the solutions of the tensor. """
    matIn = np.squeeze(matIn)
    kfwd = 0.004475761
    k4rev = 8.543317686
    k5rev = 0.12321939
    k1rev = 0.6 * 10.0 * 0.01
    k2rev = 0.6 * 144.0
    k11rev = 63.0 * k5rev / 1.5
    rxntfr = np.array([matIn[0], kfwd, k1rev, k2rev, k4rev, k5rev, k11rev,
                       matIn[6], matIn[7], matIn[8],  # IL2Ra, IL2Rb, gc
                       k1rev * 5.0, k2rev * 5.0, k4rev * 5.0, k5rev * 5.0, k11rev * 5.0])

    yOut, retVal = runCkineU_IL2(ts, rxntfr)

    assert retVal >= 0

    return yOut


def meshprep():
    """Prepares the initial conditions for the tensor."""
    # Load the data from csv file
    _, numpy_data, cell_names = import_Rexpr()
    ILs, _, _, _ = import_pstat()  # Cytokine stimulation concentrations in nM
    ILs = np.flip(ILs)

    '''Goal is to make one cell expression levels by len(mat) for every cell
    Make mesh grid of all ligand concentrations, First is IL-2 WT, Second is IL-2 Mutant; Third is IL-15.
    Set interleukins other than IL2&15 to zero. Should be of shape 3(IL2,mutIL2,IL15)*(len(ILs)) by 6 (6 for all ILs)'''
    concMesh = np.vstack((np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6),
                          np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6),
                          np.array(np.meshgrid(0, ILs, 0, 0, 0, 0)).T.reshape(-1, 6)))
    '''Repeat the cytokine stimulations (concMesh) an X amount of times where X here is number of cells (12).
    Just stacks up concMesh on top of each other 12 times (or however many cells are available)'''
    concMesh_stacked = np.tile(concMesh, (len(cell_names), 1))

    # Set receptor levels for IL7Ra, IL9R, IL4Ra, IL21Ra to one. We won't use them for IL2-15 model. Second argument can also be set to 4 since we only have IL2Ra, IL2Rb, gc, IL15Ra measured.
    no_expression = np.ones((numpy_data.shape[0], 8 - numpy_data.shape[1])) * 0.0
    # Need to convert numbers to expression values
    numpy_data[:, :] = (numpy_data[:, :] * rxntfR[17]) / (1. + ((rxntfR[20] * (1. - rxntfR[19])) / (rxntfR[21] * rxntfR[19])))  # constant according to measured number per cell
    all_receptors = np.concatenate((numpy_data, no_expression), axis=1)  # Expression: IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL4Ra, IL21Ra in order
    receptor_repeats = np.repeat(all_receptors, len(concMesh), 0)  # Create an array that repeats the receptor expression levels 'len(concMesh)' times

    '''Concatenate to obtain the new meshgrid.
    Conc_recept_cell basically repeats the initial stimulation concentration for all ligands of interest an X amount of times (X = # of cells)
    The first [len(ILs)*3] are for cell 1, second [len(ILs)*3] are for cell 2, etc...'''
    Conc_recept_cell = np.concatenate((concMesh_stacked, receptor_repeats), axis=1)
    return Conc_recept_cell, concMesh, concMesh_stacked, cell_names


def prep_tensor(numlig, n_timepoints):
    """Function to solve the model for initial conditions in meshprep()."""
    Conc_recept_cell, concMesh, concMesh_stacked, cell_names = meshprep()
    idx_ref = concMesh.shape[0] / numlig  # Provides a reference for the order of idices at which the mutant is present.

    # generate n_timepoints evenly spaced timepoints to 4 hrs
    ts = np.logspace(-3., np.log10(4 * 60.), n_timepoints)
    ts = np.insert(ts, 0, 0.0)

    # Allocate a y_of_combos
    y_of_combos = np.zeros((len(Conc_recept_cell), ts.size, nSpecies()))

    for jj in range(Conc_recept_cell.shape[0]):
        if jj % concMesh.shape[0] < idx_ref or jj % concMesh.shape[0] >= idx_ref * (numlig - 1):
            # Solve using the wildtype model.These indices are for WT IL-2 and WT IL-15.
            y_of_combos[jj] = ySolver(Conc_recept_cell[jj, :], ts)
        else:
            # Solve using the mutant model for mut-IL2
            y_of_combos[jj] = ySolver_IL2_mut(Conc_recept_cell[jj, :], ts)
    return y_of_combos, Conc_recept_cell, concMesh, concMesh_stacked, cell_names


def make_tensor(numlig=n_lig, n_timepoints=100):
    """Function to generate the 3D values tensor from the prepared solutions."""
    y_of_combos, Conc_recept_cell, concMesh, concMesh_stacked, cell_names = prep_tensor(numlig, n_timepoints)

    values = np.zeros((y_of_combos.shape[0], y_of_combos.shape[1], 1))

    values[:, :, 0] = np.tensordot(y_of_combos, getTotalActiveSpecies(), (2, 0))

    tensor3D = np.zeros((values.shape[1], len(cell_names), len(concMesh)))

    for ii in range(tensor3D.shape[0]):
        tensor3D[ii] = values[:, ii, 0].reshape(tensor3D.shape[1:3])

    return tensor3D, Conc_recept_cell, concMesh, concMesh_stacked, cell_names
