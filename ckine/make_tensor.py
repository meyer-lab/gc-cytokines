"""
Generate a tensor for the different y-values that arise at different timepoints during the model and with various initial conditions.
The initial conditions vary the concentrations of the three ligands to simulate different cell lines.
Cell lines are defined by the number of each receptor subspecies on their surface.
"""
import numpy as np
from .model import runCkineUP, getTotalActiveSpecies, receptor_expression
from .imports import import_Rexpr, import_samples_2_15, import_pstat

rxntfR, _ = import_samples_2_15(N=1, tensor=True)
rxntfR = np.squeeze(rxntfR)


# generate n_timepoints evenly spaced timepoints to 4 hrs
tensor_time = np.linspace(0.0, 240.0, 200)


def meshprep():
    """ Prepares the initial conditions for the tensor. The mutant condition here includes IL-2 mutein binding. """
    # Load the data from csv file
    _, numpy_data, cell_names = import_Rexpr()
    ILs, _, _, _, _ = import_pstat()  # Cytokine stimulation concentrations in nM
    ILs = np.flip(ILs)

    # Goal is to make one cell expression levels by len(mat) for every cell
    # Make mesh grid of all ligand concentrations, First is IL-2 WT, Third is IL-15; Fourth is IL7
    # Set interleukins other than IL2&15 to zero. Should be of shape 3(IL2,mutIL2,IL15)*(len(ILs)) by 6 (6 for all ILs)
    concMesh = np.vstack(
        (
            np.array(np.meshgrid(ILs, 0, 0, 0, 0, 0)).T.reshape(-1, 6),
            np.array(np.meshgrid(0, ILs, 0, 0, 0, 0)).T.reshape(-1, 6),
            np.array(np.meshgrid(0, 0, ILs, 0, 0, 0)).T.reshape(-1, 6),
        )
    )
    # Repeat the cytokine stimulations (concMesh) an X amount of times where X here is number of cells (12).
    # Just stacks up concMesh on top of each other 10 times (or however many cells are available)
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


def make_tensor():
    """Function to generate the 3D values tensor from the prepared solutions."""
    Conc_recept_cell, concMesh, concMesh_stacked, cell_names = meshprep()

    # Setup all the solutions
    rxn = np.tile(rxntfR.copy(), (Conc_recept_cell.shape[0], 1))
    rxn[:, 22:30] = Conc_recept_cell[:, 6:14]  # Receptor expression
    rxn[:, 0:6] = Conc_recept_cell[:, 0:6]  # Cytokine stimulation concentrations

    # Calculate solutions
    y_of_combos = runCkineUP(tensor_time, rxn)

    values = np.tensordot(y_of_combos, getTotalActiveSpecies(), (1, 0))

    tensor3D = values.reshape((-1, len(concMesh), len(cell_names)), order='F')
    tensor3D = np.swapaxes(tensor3D, 1, 2)

    return tensor3D, Conc_recept_cell, concMesh, concMesh_stacked, cell_names
