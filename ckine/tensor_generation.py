"""
Generate a tensor for the different y-values that arise at different timepoints during the model and with various initial conditions. The initial conditions vary the concentrations of the ligands and the expression rates of the receptors to simulate different cell lines.
"""
import numpy as np
from tqdm import tqdm
from scipy.integrate import odeint
from .model import solveAutocrine, fullModel, getTotalActiveCytokine, __active_species_IDX, surfaceReceptors, totalReceptors


def findy(lig, exp):
    """A function to find the different values of y at different timepoints and different initial conditions. Takes in how many ligand concentrations and expression rates to iterate over."""
    t = 60. * 4 # let's let the system run for 4 hours
    ts = np.linspace(0.0, t, 100) #generate 100 evenly spaced timepoints
    IL2 = IL15 = IL7 = IL9 = np.logspace(-3, 3, num=lig)
    IL2Ra = IL2Rb = gc = IL15Ra = IL7Ra = IL9R = np.logspace(-3, 2, num=exp)
    mat = np.array(np.meshgrid(IL2,IL15,IL7,IL9,IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R)).T.reshape(-1, 10)
    #print (mat.shape[0]) gives 1024 for the above values; Need to update according to choice

    y_of_combos = np.zeros((len(mat), 100,56))

    #Set some given parameters already determined from fitting
    r = np.zeros(17)
    r[4:17] = np.ones(13) * (5*10**-1)   #I am supposed to have these values

    trafRates = np.zeros(11)
    trafRates[0:5] = (5* 10**-2)

    #Iterate through every combination of values and store odeint values in a y matrix
    for ii in tqdm(range(len(mat))):
        #Create a new y0 everytime odeint is run per combination of values.
        trafRates[5:8], trafRates[8], trafRates[9], trafRates[10] = mat[ii,4:7], mat[ii,7], mat[ii,8], mat[ii,9]
        y0 = solveAutocrine(trafRates)
        r[0:4] = mat[ii,0:4]
        #Running odeint gives y for each of the 100 timepoints.
        ddfunc = lambda y, t: fullModel(y, t, r, trafRates, __active_species_IDX)
        temp, d = odeint(ddfunc, y0, ts, mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        if d['message'] == "Integration successful.":
            y_of_combos[ii] = temp # only assign values to ys if there isn't an error message; all errors will still be 0
    return y_of_combos, mat


def activity_surface_total(yVec):
    """This function returns a vector of 16 elements where the activity of the 4 cytokines and amounts of surface and total receptors are included."""
    x = np.zeros(16)
    x[0],x[1],x[2],x[3] = getTotalActiveCytokine(0,yVec), getTotalActiveCytokine(1,yVec), getTotalActiveCytokine(2,yVec), getTotalActiveCytokine(3,yVec)
    x[4:10] = surfaceReceptors(yVec)
    x[10:16] = totalReceptors(yVec)
    return x

def activity_surf_tot(y_of_combos):
    """This function returns the activity and amounts of receptors both on the surface and total for every timepoint per combination of values"""
    values = np.zeros((len(y_of_combos), 100,16))

    for i, _ in enumerate(y_of_combos):
        for j in range(100):
            values[i][j] = activity_surface_total(y_of_combos[i][j])
    return values

"""Important Notes:
    y_of_combos is a multidimentional matrix of size (length mesh x 100 timeponts x 56 values of y)
    values is also a multidementinal matrix of size (length mesh x 100 x 16 values for cytokine activity, surface receptors amount, and total receptors amount)
"""
