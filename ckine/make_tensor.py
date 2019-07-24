"""
Generate a tensor for the different y-values that arise at different timepoints during the model and with various initial conditions.
The initial conditions vary the concentrations of the three ligands to simulate different cell lines.
Cell lines are defined by the number of each receptor subspecies on their surface.
"""
import numpy as np
from .model import runCkineU, nSpecies, runCkineU_IL2, getTotalActiveSpecies, receptor_expression
from .imports import import_Rexpr, import_samples_2_15, import_pstat

rxntfR, _ = import_samples_2_15(N=1, tensor=True)
rxntfR = np.squeeze(rxntfR)


# generate n_timepoints evenly spaced timepoints to 4 hrs
tensor_time = np.linspace(0., 240., 200)

def n_lig(mut):
    '''Function to return the number of cytokines used in building the tensor.'''
    #Mutant here refers to a tensor made exclusively of WT IL-2 and mutant affinity IL-2.
    if mut:
        nlig = 3
    else:
        nlig = 4
    return nlig

def ySolver(matIn, ts, tensor=True):
    """ This generates all the solutions for the Wild Type interleukins across conditions defined in meshprep(). """
    matIn = np.squeeze(matIn)
    rxn = rxntfR.copy()

    if tensor:
        rxn[22:30] = matIn[6:14]  # Receptor expression
    rxn[0:6] = matIn[0:6]  # Cytokine stimulation concentrations in the following order: IL2, 15, 7, 9, 4, 21, and in nM

    temp, retVal = runCkineU(ts, rxn)
    assert retVal >= 0

    return temp


def ySolver_IL2_mut(matIn, ts, mut):
    """ This generates all the solutions of the tensor. """
    matIn = np.squeeze(matIn).copy()
    kfwd, k4rev, k5rev = rxntfR[6], rxntfR[7], rxntfR[8]
    k1rev = 0.6 * 10.0
    k2rev = 0.6 * 144.0
    k11rev = 63.0 * k5rev / 1.5

    if mut == 'a':
        k2rev *= 10.0  # 10x weaker binding to IL2Rb
    elif mut == 'b':
        k2rev *= 0.01  # 100x more bindng to IL2Rb

    rxntfr = np.array([matIn[0], kfwd, k1rev, k2rev, k4rev, k5rev, k11rev,
                       matIn[6], matIn[7], matIn[8],  # IL2Ra, IL2Rb, gc
                       k1rev * 5.0, k2rev * 5.0, k4rev * 5.0, k5rev * 5.0, k11rev * 5.0])

    yOut, retVal = runCkineU_IL2(ts, rxntfr)

    assert retVal >= 0

    return yOut


def meshprep(mut):
    """ Prepares the initial conditions for the tensor. The mutant condition here includes IL-2 mutein binding. """
    # Load the data from csv file
    _, numpy_data, cell_names = import_Rexpr()
    ILs, _, _, _, _, _ = import_pstat()  # Cytokine stimulation concentrations in nM
    ILs = np.flip(ILs)

    # Goal is to make one cell expression levels by len(mat) for every cell
    # Make mesh grid of all ligand concentrations, First is IL-2 WT, Third is IL-15; Fourth is IL7
    # Set interleukins other than IL2&15 to zero. Should be of shape 3(IL2,mutIL2,IL15)*(len(ILs)) by 6 (6 for all ILs)
    if mut:
        concMesh = np.vstack((np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6),
                              np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6),
                              np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6)))
    else:
        concMesh = np.vstack((np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6),
                              np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6),
                              np.array(np.meshgrid(0, ILs, 0, 0, 0, 0)).T.reshape(-1, 6),
                              np.array(np.meshgrid(0, 0, ILs, 0, 0, 0)).T.reshape(-1, 6)))
    # Repeat the cytokine stimulations (concMesh) an X amount of times where X here is number of cells (12).
    # Just stacks up concMesh on top of each other 12 times (or however many cells are available)
    concMesh_stacked = np.tile(concMesh, (len(cell_names), 1))

    # Set receptor levels for IL9R, IL4Ra, IL21Ra to 0. We won't use them for IL2-15 model. Second argument can also be set to 4 since we only have IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra measured.
    no_expression = np.ones((numpy_data.shape[0], 8 - numpy_data.shape[1])) * 0.0
    # Need to convert numbers to expression values
    numpy_data[:, :] = receptor_expression(numpy_data[:, :], rxntfR[17], rxntfR[20], rxntfR[19], rxntfR[21])

    # Expression: IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL4Ra, IL21Ra in order
    all_receptors = np.concatenate((numpy_data, no_expression), axis=1)
    # Create an array that repeats the receptor expression levels 'len(concMesh)' times
    receptor_repeats = np.repeat(all_receptors, len(concMesh), 0)

    # Concatenate to obtain the new meshgrid.
    # Conc_recept_cell repeats the initial stimulation concentration for all ligands of interest X times (X = # of cells)
    # The first [len(ILs)*4] are for cell 1, second [len(ILs)*4] are for cell 2, etc...
    Conc_recept_cell = np.concatenate((concMesh_stacked, receptor_repeats), axis=1)
    return Conc_recept_cell, concMesh, concMesh_stacked, cell_names


def prep_tensor(mut):
    """Function to solve the model for initial conditions in meshprep()."""
    Conc_recept_cell, concMesh, concMesh_stacked, cell_names = meshprep(mut)
    numlig = n_lig(mut)
    idx_ref = int(concMesh.shape[0] / numlig)  # Provides a reference for the order of indices at which the mutant is present.

    # Allocate a y_of_combos
    y_of_combos = np.zeros((len(Conc_recept_cell), tensor_time.size, nSpecies()))

    if mut:
        mut2 = np.arange(0, Conc_recept_cell.shape[0], idx_ref)
        IL2Ra = mut2[np.arange(1, mut2.size, numlig)]
        IL2Rb = mut2[np.arange(2, mut2.size, numlig)]
        IL2Ra_idxs = np.zeros((IL2Ra.size, idx_ref))
        IL2Rb_idxs = IL2Ra_idxs.copy()
        for jj, _ in enumerate(IL2Ra):
            IL2Ra_idxs[jj] = np.array(range(IL2Ra[jj], IL2Ra[jj] + idx_ref))  # Find the indices where the IL2-mutant is.
            IL2Rb_idxs[jj] = np.array(range(IL2Rb[jj], IL2Rb[jj] + idx_ref))

        for jj, row in enumerate(Conc_recept_cell):
            if jj in IL2Ra_idxs:
                y_of_combos[jj] = ySolver_IL2_mut(row, tensor_time, mut='a')  # Solve using the mutant IL2-IL2Ra solver
            elif jj in IL2Rb_idxs:
                y_of_combos[jj] = ySolver_IL2_mut(row, tensor_time, mut='b')  # Solve using the mutant IL2-IL2Rb solver
            else:
                y_of_combos[jj] = ySolver(row, tensor_time)  # Solve using the WT solver for IL2.
    else:
        mut2 = np.arange(0, Conc_recept_cell.shape[0], idx_ref)
        IL2Ra = mut2[np.arange(1, mut2.size, numlig)]
        IL2Ra_idxs = np.zeros((IL2Ra.size, idx_ref))
        for jj, _ in enumerate(IL2Ra):
            IL2Ra_idxs[jj] = np.array(range(IL2Ra[jj], IL2Ra[jj] + idx_ref))  # Find the indices where the IL2-mutant is.

        for jj, row in enumerate(Conc_recept_cell):
            # Solve using the WT solver for each of IL2, IL15, and IL7. And the mutant Solver for IL-2--Il-2Ra.
            if jj in IL2Ra_idxs:
                y_of_combos[jj] = ySolver_IL2_mut(row, tensor_time, mut='a')  # Solve using the mutant IL2-IL2Ra solver
            else:
                y_of_combos[jj] = ySolver(row, tensor_time)

    return y_of_combos, Conc_recept_cell, concMesh, concMesh_stacked, cell_names


def make_tensor(mut=False):
    """Function to generate the 3D values tensor from the prepared solutions."""
    y_of_combos, Conc_recept_cell, concMesh, concMesh_stacked, cell_names = prep_tensor(mut)

    values = np.zeros((y_of_combos.shape[0], y_of_combos.shape[1], 1))

    values[:, :, 0] = np.tensordot(y_of_combos, getTotalActiveSpecies(), (2, 0))

    tensor3D = np.zeros((values.shape[1], len(cell_names), len(concMesh)))

    for ii in range(tensor3D.shape[0]):
        tensor3D[ii] = values[:, ii, 0].reshape(tensor3D.shape[1:3])

    return tensor3D, Conc_recept_cell, concMesh, concMesh_stacked, cell_names
