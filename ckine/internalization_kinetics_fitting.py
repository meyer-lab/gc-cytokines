from .model import fullModel, __active_species_IDX, printModel
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
          
    # find the percent of initial IL2Rb at the cell surface for the time points mentioned in the figure and compare to observed values
    # this function handles the case of IL2 = 1 nM and IL2Ra+
    def surf_IL2Rb_1(self, y0, rxnRates, trafRates):
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
        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
        
        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
        return percent_surface_IL2Rb - self.numpy_data[1:8, 1] # the second column of numpy_data has all the 1nM IL2 data
        
    # this function handles the case of IL2 = 500 nM and IL2Ra+
    def surf_IL2Rb_2(self, y0, rxnRates, trafRates):
        rxnRates[0] = 500. # the concentration of IL2 = 1 nM
        
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
        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
        
        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
        return percent_surface_IL2Rb - self.numpy_data[1:8, 5] # the sixth column of numpy_data has all the 500 nM IL2 data

# this function handles the case of IL2 = 1 nM and IL2Ra-
    def surf_IL2Rb_3(self, y0, rxnRates, trafRates):
        rxnRates[0] = 1. # the concentration of IL2 = 1 nM
        
        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
        ts = self.numpy_data2[1:8, 0] # the first column has the time data
        ys = np.zeros(8, 52) 
        
        for ii in range(0,8):
            
            ys[ii, :], infodict = odeint(ddfunc, y0, ts[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        
            if infodict['tcur'] < np.max(ts):
                # print("IL2 conc: " + str(IL2))
                printModel(rxnRates, trafRates)
                print(infodict)
                return -100
        
        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
        
        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
        return percent_surface_IL2Rb - self.numpy_data2[1:8, 1] # the second column of numpy_data has all the 1nM IL2 data

    # this function handles the case of IL2 = 500 nM and IL2Ra-
    def surf_IL2Rb_4(self, y0, rxnRates, trafRates):
        rxnRates[0] = 500. # the concentration of IL2 = 1 nM
        
        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
        ts = self.numpy_data2[1:8, 0] # the first column has the time data... might have a problem with the header being a string
        ys = np.zeros(8, 52) 
        
        for ii in range(0,8):
            
            ys[ii, :], infodict = odeint(ddfunc, y0, ts[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        
            if infodict['tcur'] < np.max(ts):
                # print("IL2 conc: " + str(IL2))
                printModel(rxnRates, trafRates)
                print(infodict)
                return -100
        
        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
        
        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
        return percent_surface_IL2Rb - self.numpy_data2[1:8, 5] # the sixth column of numpy_data has all the 500 nM IL2 data



class build_model:
    
    # going to load the data from the CSV file at the very beginning of when build_model is called... needs to be separate member function to avoid uploading file thousands of times
    def __init__(self):
        self.IL2Rb = IL2Rb_trafficking()
        self.M = self.build()
    
    def build(self):
        M = pm.Model()

        with M:
            rxnrates = pm.Lognormal('rxn', sd=1., shape=3, testval=[0.1, 0.1, 0.1]) 
            endo_activeEndo = pm.Lognormal('endo', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            kRec_kDeg = pm.Lognormal('kRec_kDeg', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            Rexpr = pm.Lognormal('IL2Raexpr', sd=1., shape=3, testval=[1., 1., 1.])
            sortF = pm.Beta('sortF', alpha=2, beta=7, testval=0.1)

            unkVec = T.concatenate((rxnrates, endo_activeEndo, T.stack(sortF), kRec_kDeg, Rexpr))
            
            Y = centralDiff(self.IL2Rb)(unkVec) # fitting the data based on dst.calc for the given parameters
            
            pm.Deterministic('Y', Y) # this line allows us to see the traceplots in read_fit_data.py... it lets us know if the fitting process is working

            pm.Normal('fitD', sd=0.1, observed=Y) # TODO: Find an empirical value for the SEM

            # Save likelihood
            pm.Deterministic('logp', M.logpt)

        return M
    
    def sampling(self):
        with self.M:
            try:
                self.trace = pm.sample()
            except ValueError:
                # Something went wrong, so print out the variables.
                print("Test point:")
                point = self.M.test_point
                logp = self.M.logp
                dlogp = self.M.dlogp()

                print(point)
                print(logp(point))
                print(dlogp(point))

                raise

    def profile(self):
        """ Profile the gradient calculation. """
        self.M.profile(pm.theanof.gradient(self.M.logpt, None)).summary()