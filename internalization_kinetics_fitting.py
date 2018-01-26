from ckine.model import fullModel, __active_species_IDX, printModel, solveAutocrine
from scipy.integrate import odeint
import numpy as np, pandas as pds
from ckine.differencing_op import centralDiff
import pymc3 as pm, theano.tensor as T, os
from ckine.fit import IL2_convertRates

def surf_IL2Rb_1(unkVec):
        rxnRates, trafRates = IL2_convertRates(unkVec) # this function splits up unkVec into rxnRates and trafRates
        y0 = solveAutocrine(trafRates) # solveAutocrine in model.py gives us the y0 values based on trafRates
        
        rxnRates500 = rxnRates.copy()
        rxnRates[0] = 1. # the concentration of IL2 = 1 nM
        rxnRates500[0] = 500. # the concentration of IL2 = 500 nM
               
        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
        ddfunc500 = lambda y, t: fullModel(y, t, rxnRates500, trafRates, __active_species_IDX)
        
        ys = np.zeros((7, 56)) 
        ys500 = ys.copy()

        # times from experiment are hard-coded into this function      
        ts = np.array(([4.26260000e-02, 2.14613600e+00, 5.00755700e+00, 1.52723300e+01, 3.07401000e+01, 6.14266000e+01, 9.21962300e+01]))
        
        ys[:,:], infodict = odeint(ddfunc, y0, ts, mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        ys500[:,:], infodict = odeint(ddfunc500, y0, ts, mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
        
#        if infodict['tcur'] < np.max(self.ts):
#            # print("IL2 conc: " + str(IL2))
#            printModel(rxnRates, trafRates)
#            print(infodict)
#            return -100
        
        print(ys[:,1]) # getting inf and nan as first two outputs of IL2Ra- case; IL2Ra+ cases make sense
        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
        
        surface_IL2Rb_500 = ys500[:,1]
        initial_surface_IL2Rb_500 = surface_IL2Rb_500[0]
        
        
        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
        percent_surface_IL2Rb_500 = 10. * (surface_IL2Rb_500 / initial_surface_IL2Rb_500)
        
        percent_surface_IL2Rb_total = np.concatenate((percent_surface_IL2Rb, percent_surface_IL2Rb_500))
        
#        print(percent_surface_IL2Rb_total)
        # the first time this function is called in calc_schedule (unkVec) it produces all 10's
        # the second time this function is called (unkVec2) it produces all 'nan's
        print(percent_surface_IL2Rb_total) # works for IL2Ra+ cases but breaks down for both IL2Ra- cases
        
        return percent_surface_IL2Rb_total


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

    
        
#    # this function handles the case of IL2 = 500 nM and IL2Ra+
#    def surf_IL2Rb_2(self, unkVec):
#        rxnRates, trafRates = IL2_convertRates(unkVec) # this function splits up unkVec into rxnRates and trafRates
#        y0 = solveAutocrine(trafRates) # solveAutocrine in model.py gives us the y0 values based on trafRates
#        
#        rxnRates[0] = 500. # the concentration of IL2 = 1 nM
#        
#        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
#        ys = np.zeros((7, 56)) 
#        
#        for ii in range(0,7):
#            
#            ys[ii, :], infodict = odeint(ddfunc, y0, self.ts[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
#        
#            if infodict['tcur'] < np.max(self.ts):
#                # print("IL2 conc: " + str(IL2))
#                printModel(rxnRates, trafRates)
#                print(infodict)
#                return -100
#        
#        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
#        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
#        
#        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
#        return percent_surface_IL2Rb 
#
## this function handles the case of IL2 = 1 nM and IL2Ra-
#    def surf_IL2Rb_3(self, unkVec):
#        rxnRates, trafRates = IL2_convertRates(unkVec) # this function splits up unkVec into rxnRates and trafRates
#        y0 = solveAutocrine(trafRates) # solveAutocrine in model.py gives us the y0 values based on trafRates
#        
#        rxnRates[0] = 1. # the concentration of IL2 = 1 nM
#        
#        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
#        ys = np.zeros((7, 56)) 
#        
#        for ii in range(0,7):
#            
#            ys[ii, :], infodict = odeint(ddfunc, y0, self.ts2[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
#        
#            if infodict['tcur'] < np.max(self.ts2):
#                # print("IL2 conc: " + str(IL2))
#                printModel(rxnRates, trafRates)
#                print(infodict)
#                return -100
#        
#        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
#        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
#        
#        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
#        return percent_surface_IL2Rb 
#
#    # this function handles the case of IL2 = 500 nM and IL2Ra-
#    def surf_IL2Rb_4(self, unkVec):
#        rxnRates, trafRates = IL2_convertRates(unkVec) # this function splits up unkVec into rxnRates and trafRates
#        y0 = solveAutocrine(trafRates) # solveAutocrine in model.py gives us the y0 values based on trafRates
#        
#        rxnRates[0] = 500. # the concentration of IL2 = 1 nM
#        
#        ddfunc = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)
#        ys = np.zeros((7, 56)) 
#        
#        for ii in range(0,7):
#            
#            ys[ii, :], infodict = odeint(ddfunc, y0, self.ts2[ii], mxstep=12000, full_output=True, rtol=1.0E-5, atol=1.0E-3)
#        
#            if infodict['tcur'] < np.max(self.ts2):
#                # print("IL2 conc: " + str(IL2))
#                printModel(rxnRates, trafRates)
#                print(infodict)
#                return -100
#        
#        surface_IL2Rb = ys[:,1] # y[:,1] represents the surface IL2Rb value in fullModel for all 8 time points
#        initial_surface_IL2Rb = surface_IL2Rb[0] # find the total amount of IL2Rb in the system at the first time point
#        
#        percent_surface_IL2Rb = 10. * (surface_IL2Rb / initial_surface_IL2Rb) # percent of surface IL2Rb is relative to the initial amount of receptor
#        return percent_surface_IL2Rb

    def calc_schedule(self, unkVec, pool):
        # Convert the vector of values to dicts
        rxnRates, tfR = IL2_convertRates(unkVec)

        # IL2Ra- cells
        tfR2 = tfR.copy()
        tfR2[5] = 0.0

        unkVec2 = np.concatenate((rxnRates, tfR2))
        # Loop over concentrations of IL2
        output = list()
        
        # do I need to change the ILc variable too? I already changed the self.IL2s to self.ts
        output.append(pool.submit(surf_IL2Rb_1, unkVec)) # handle the IL2Ra+ case

        output.append(pool.submit(surf_IL2Rb_1, unkVec2)) # then handle the IL2Ra- case
        
        return output
    
    
    def calc_reduce(self, inT):
        output = inT
        
        actVec = list(item.result() for item in output) # changed count to self.times
        
        print('actVec: ' + str(actVec))
        # for some reason I'm getting "nan" for all 14 of my values corresponding to the IL2Ra- case
        # also every value for the IL2Ra+ cases is 10 which might be a problem because they're supposed to be time-series values run by odeint
        
        
        # actVec[0:7] represents the IL2Ra+ and 1nM case
        # actVec[7:14] represents the IL2Ra+ and 500nM case
        # actVec[14:21] represents the IL2Ra- and 1nM case
        # actVec[21:28] represents the IL2Ra- and 500nM case
        diff = actVec[0:7] - self.numpy_data[:, 1] # the second column of numpy_data has all the 1nM IL2Ra= data
        diff2 = actVec[7:14] - self.numpy_data[:, 5] # the sixth column of numpy_data has all the 500 nM IL2Ra+ data
        diff3 = actVec[14:21] - self.numpy_data2[:, 1] # the second column of numpy_data has all the 1nM IL2Ra- data
        diff4 = actVec[21:28] - self.numpy_data2[:, 5] # the sixth column of numpy_data has all the 500 nM IL2Ra- data
        
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
