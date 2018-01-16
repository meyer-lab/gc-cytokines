from .model import solveAutocrine, fullModel, getTotalActiveCytokine, __active_species_IDX, printModel
from scipy.integrate import odeint
import numpy as np, pandas as pds
from .differencing_op import centralDiff
import pymc3 as pm, theano.tensor as T, os

class IL2Rb_trafficking:
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(os.path.join(path, "./data/IL2Ra+_surface_IL2RB_datasets.csv")) # imports csv file into pandas array
        self.numpy_data = data.as_matrix() # all of the IL2Rb trafficking data with IL2Ra+... first row contains headers... 9 columns and 8 rows... first column is time
        data2 = pds.read_csv(os.path.join(path, "./data/IL2Ra-_surface_IL2RB_datasets.csv"))
        self.numpy_data2 = data2.as_matrix() # all of the IL2Rb trafficking data with IL2Ra-... first row contains headers... 9 columns and 8 rows... first column is time
        
    
    # find the amount of IL2Rb at the cell surface for the time points mentioned in the figure
    # do this for just the 1 nM of IL2 and IL2Ra+ case for now... can add 500 nM and IL2Ra- later
    def perc_surf_IL2Rb(self, y0, rxnRates, trafRates):
        rxnRates[0] = 1. # the concentration of IL2 = 1 nM
        
        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
        ts = self.numpy_data[1:8, 0] # the first column has the time data... might have a problem with the header being a string
        ys = np.zeros(8, 52) 
        
        for ii in range(0,8):
            
            ys[ii, :], infodict = odeint(ddfunc, y0, ts[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        
            if infodict['tcur'] < np.max(ts):
                # print("IL2 conc: " + str(IL2))
                printModel(rxnRates, trafRates)
                print(infodict)
                return -100
        
        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
        endo_IL2Rb = ys[:, 27] 
        total_IL2Rb = surface_IL2Rb + endo_IL2Rb
        
        percent_surface_IL2Rb = 100. * (surface_IL2Rb / total_IL2Rb)
        return percent_surface_IL2Rb
        
    # find the difference between the %*10 surface IL2Rb values in the figure and from the above calculations
    def compare_to_figure(self, y0, rxnRates, trafRates):
        # might have to transpose the matrix with the experimental data
        percent_surface_IL2Rb = perc_surf_IL2Rb(y0, rxnRates, trafRates)
        
        diff = self.numpy_data - (percent_surface_IL2Rb * 10.)
        return diff
