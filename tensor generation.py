from ckine.model import solveAutocrine, fullModel, getTotalActiveCytokine, __active_species_IDX, printModel, getActiveCytokine
import numpy as np
from scipy.integrate import odeint

t = 100000. # let's let the system run for 100000
ts = np.linspace(0.0, t, 100)

#Different indices of important values
#y0 = np.zeros(56)
#y0[0:3], y0[10], y0[18], y0[22], y0[52:56] = IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL2, IL15, IL7, IL9


IL2 = IL15 = IL7 = IL9 = np.logspace(-3, 3, num=2)
IL2Ra = IL2Rb = gc = IL15Ra = IL7Ra = IL9R = np.logspace(-3, 2, num=2)
mat = np.array(np.meshgrid(IL2,IL15,IL7,IL9,IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R)).T.reshape(-1, 10)
#print (mat.shape[0]) gives 1024 for the above values; Need to update according to choice

y_of_combos = np.zeros((len(mat), 100,56))

#Set some given parameters already determined from fitting
r = np.zeros(17)
r[4:17] = np.ones(13) * (5*10**-1)   #I am supposed to have these values
tfR = np.ones(17) * (5* 10**-2)     #I am also supposed to know these values

#Iterate through every combination of values and store odeint values in a y matrix
for ii in range(len(mat)+1):
    y0 = np.zeros(56)
    
    y0[0:3], y0[10], y0[18], y0[22] = mat[ii,4:7], mat[ii,7], mat[ii,8], mat[ii,9] 
    r[0:4] = mat[ii,0:4]
    ddfunc = lambda y, t: fullModel(y, t, r, tfR, __active_species_IDX)
    
    temp, d = odeint(ddfunc, y0, ts, mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
    #temp, d = odeint(fullModel, y0, ts, r, tfR, __active_species_IDX, mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
    if d['message'] == "Integration successful.":
        y_of_combos[ii] = temp # only assign values to ys if there isn't an error message; all errors will still be 0

def sum_surface_recep(y):
    ans = y[]


def activ_surf_tot(y_of_combos):
    lig_activity = np.zeros((len(mat), 100,4))
    surface_receptors = np.zeros((len(mat), 100,6))

    for i in range(len(mat) +1):
        for j in range(7):
            for k in range(len(ts)+1):
                if j <= 3:
                    lig_activity[i][:,j][k] = getActiveCytokine(j, y_of_combos[i][k])
                    surface_receptors[i][:,j][k] = sum_surface_recep
                    
                    #####
                    #####
                    #####
                    ####
                    ####
                    ####
                    #Continue working on this
                
                
                



