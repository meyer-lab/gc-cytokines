from ckine.model import fullModel, __active_species_IDX, printModel, solveAutocrine
from scipy.integrate import odeint
import numpy as np, pandas as pds
from ckine.differencing_op import centralDiff
import pymc3 as pm, theano.tensor as T, os
from ckine.fit import IL2_convertRates

class IL2Rb_trafficking:
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        data = pds.read_csv(os.path.join(path, "ckine/data/IL2Ra+_surface_IL2RB_datasets.csv")) # imports csv file into pandas array
        self.numpy_data = data.as_matrix() # all of the IL2Rb trafficking data with IL2Ra+... first row contains headers... 9 columns and 8 rows... first column is time
        data2 = pds.read_csv(os.path.join(path, "ckine/data/IL2Ra-_surface_IL2RB_datasets.csv"))
        self.numpy_data2 = data2.as_matrix() # all of the IL2Rb trafficking data with IL2Ra-... first row contains headers... 9 columns and 8 rows... first column is time
        self.ts = self.numpy_data[:, 0]
        self.ts2 = self.numpy_data2[:, 0]
        self.times = len(self.ts)
#        print(self.ts.shape)
#        print(self.numpy_data[0,0])
        
    # find the percent of initial IL2Rb at the cell surface for the time points mentioned in the figure and compare to observed values
    # this function handles the case of IL2 = 1 nM and IL2Ra+
    def surf_IL2Rb_1(self, unkVec):
        rxnRates, trafRates = IL2_convertRates(unkVec) # this function splits up unkVec into rxnRates and trafRates
        y0 = solveAutocrine(trafRates) # solveAutocrine in model.py gives us the y0 values based on trafRates
        
        rxnRates[0] = 1. # the concentration of IL2 = 1 nM
        
        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
        ys = np.zeros((7, 56)) 
#        print(ys.shape)
#        
#        print(self.ts.shape)
        for ii in range(0,7):
            
            ys[ii, :], infodict = odeint(ddfunc, y0, self.ts[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        
            if infodict['tcur'] < np.max(self.ts):
                # print("IL2 conc: " + str(IL2))
                printModel(rxnRates, trafRates)
                print(infodict)
                return -100
        
        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
        
        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
        return percent_surface_IL2Rb 
        
    # this function handles the case of IL2 = 500 nM and IL2Ra+
    def surf_IL2Rb_2(self, unkVec):
        rxnRates, trafRates = IL2_convertRates(unkVec) # this function splits up unkVec into rxnRates and trafRates
        y0 = solveAutocrine(trafRates) # solveAutocrine in model.py gives us the y0 values based on trafRates
        
        rxnRates[0] = 500. # the concentration of IL2 = 1 nM
        
        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
        ys = np.zeros((7, 56)) 
        
        for ii in range(0,7):
            
            ys[ii, :], infodict = odeint(ddfunc, y0, self.ts[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        
            if infodict['tcur'] < np.max(self.ts):
                # print("IL2 conc: " + str(IL2))
                printModel(rxnRates, trafRates)
                print(infodict)
                return -100
        
        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
        
        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
        return percent_surface_IL2Rb 

# this function handles the case of IL2 = 1 nM and IL2Ra-
    def surf_IL2Rb_3(self, unkVec):
        rxnRates, trafRates = IL2_convertRates(unkVec) # this function splits up unkVec into rxnRates and trafRates
        y0 = solveAutocrine(trafRates) # solveAutocrine in model.py gives us the y0 values based on trafRates
        
        rxnRates[0] = 1. # the concentration of IL2 = 1 nM
        
        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
        ys = np.zeros((7, 56)) 
        
        for ii in range(0,7):
            
            ys[ii, :], infodict = odeint(ddfunc, y0, self.ts2[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        
            if infodict['tcur'] < np.max(self.ts2):
                # print("IL2 conc: " + str(IL2))
                printModel(rxnRates, trafRates)
                print(infodict)
                return -100
        
        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
        
        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
        return percent_surface_IL2Rb 

    # this function handles the case of IL2 = 500 nM and IL2Ra-
    def surf_IL2Rb_4(self, unkVec):
        rxnRates, trafRates = IL2_convertRates(unkVec) # this function splits up unkVec into rxnRates and trafRates
        y0 = solveAutocrine(trafRates) # solveAutocrine in model.py gives us the y0 values based on trafRates
        
        rxnRates[0] = 500. # the concentration of IL2 = 1 nM
        
        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
        ys = np.zeros((7, 56)) 
        
        for ii in range(0,7):
            
            ys[ii, :], infodict = odeint(ddfunc, y0, self.ts2[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        
            if infodict['tcur'] < np.max(self.ts2):
                # print("IL2 conc: " + str(IL2))
                printModel(rxnRates, trafRates)
                print(infodict)
                return -100
        
        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
        
        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
        return percent_surface_IL2Rb

    def calc_schedule(self, unkVec, pool):
        # Loop over concentrations of IL2
        output = list()
        
        # do I need to change the ILc variable too? I already changed the self.IL2s to self.ts
        output.append(pool.submit(self.surf_IL2Rb_1, unkVec))

        output.append(pool.submit(self.surf_IL2Rb_2, unkVec))

        output.append(pool.submit(self.surf_IL2Rb_3, unkVec))

        output.append(pool.submit(self.surf_IL2Rb_4, unkVec))
        
        return output
    
    
    def calc_reduce(self, inT):
        output = inT
        
        actVec = list(item.result() for item in output) # changed count to self.times

        
        diff = actVec[0] - self.numpy_data[:, 1] # the second column of numpy_data has all the 1nM IL2 data
        diff2 = actVec[1] - self.numpy_data[:, 5] # the sixth column of numpy_data has all the 500 nM IL2 data
        diff3 = actVec[2] - self.numpy_data2[:, 1] # the second column of numpy_data has all the 1nM IL2 data
        diff4 = actVec[3] - self.numpy_data2[:, 5] # the sixth column of numpy_data has all the 500 nM IL2 data
        
        all_diffs = np.concatenate((diff, diff2, diff3, diff4))
        
        return all_diffs
        
        
    def calc(self, unkVec, pool):
        """ Just get the solution in one pass. """
        inT = self.calc_schedule(unkVec, pool)
        return self.calc_reduce(inT)

class build_model:
    
    # going to load the data from the CSV file at the very beginning of when build_model is called... needs to be separate member function to avoid uploading file thousands of times
    def __init__(self):
        self.IL2Rb = IL2Rb_trafficking()
        self.M = self.build()
    
    def build(self):
        M = pm.Model()

        with M:
            rxnrates = pm.Lognormal('rxn', sd=1., shape=3, testval=[0.1, 0.1, 0.1]) # number of unknowns reaction rates
            endo_activeEndo = pm.Lognormal('endo', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            kRec_kDeg = pm.Lognormal('kRec_kDeg', mu=np.log(0.1), sd=1., shape=2, testval=[0.1, 0.1])
            Rexpr = pm.Lognormal('IL2Raexpr', sd=1., shape=3, testval=[1., 1., 1.]) # IL2Ra, IL2Rb, gc (shape of 3)
            sortF = pm.Beta('sortF', alpha=2, beta=7, testval=0.1)

            unkVec = T.concatenate((rxnrates, endo_activeEndo, T.stack(sortF), kRec_kDeg, Rexpr))
            
            Y = centralDiff(self.IL2Rb)(unkVec) # fitting the data based on IL2Rb_trafficking class
            
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
        
build_model()
