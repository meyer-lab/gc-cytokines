"""File to simulate model with different drug combinations, used for plotting."""
import numpy as np
from .model import nSpecies, getTotalActiveCytokine
from .tensor_generation import ySolver


def input_IL2_15(final_conc, num):
    '''Function that creates the input for the solver. Takes in 1nM or 500nM for final_conc.'''
    Rexpr = np.array([3.8704, 0.734, 1.7147, 0.32010875, 0.0, 0.0, 0.0, 0.0])  # Il2ra, Il2rb, Il2rg, Il15ra, Il7r, Il9r, IL4Ra, IL21Ra in that order
    ligand_conc = np.zeros((num, 6))  # Cytokine stimulation concentrations in the following order: IL2, 15, 7, 9, 4, 21, and in nM.
    xaxis = np.linspace(-6, 6, num)
    for idx, C_2 in enumerate(xaxis):
        A = np.array([[1, 1], [1, -10**C_2]])
        B = np.array([final_conc, 0])
        C = np.linalg.solve(A, B)
        ligand_conc[idx, 0:2] = [C[0], C[1]]
    return ligand_conc, Rexpr, xaxis


def solve_IL2_IL15(final_conc, num, time, nSpecies=nSpecies):
    """Function to simulate model with IL2 and IL15 only at timepoint tps."""
    ligand_conc, Rexpr, xaxis = input_IL2_15(final_conc, num)
    tps = np.array([time])
    yOut = np.zeros((num, nSpecies()))
    active = np.zeros((3, num))  # First row is IL2 activity alone, second row is IL15 activity alone, third row is activity sum (total)
    for ii in range(num):
        yOut[ii] = ySolver(np.concatenate((ligand_conc[ii], Rexpr)), tps)
        active[0, ii] = getTotalActiveCytokine(0, np.squeeze(yOut[ii]))
        active[1, ii] = getTotalActiveCytokine(1, np.squeeze(yOut[ii]))
    active[2] = active[0] + active[1]
    return active, num, xaxis
