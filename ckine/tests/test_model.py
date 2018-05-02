"""
Unit test file.
"""
import unittest
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays as harrays
from ..model import dy_dt, fullModel, solveAutocrine, getTotalActiveCytokine, solveAutocrineComplete, runCkine, jacobian
from ..util_analysis.Shuffle_ODE import approx_jacobian, approx_jac_dydt
from ..Tensor_analysis import find_R2X

settings.register_profile("ci", max_examples=1000)
#settings.load_profile("ci")

class TestModel(unittest.TestCase):
    def assertPosEquilibrium(self, X, func):
        """Assert that all species came to equilibrium."""
        # All the species abundances should be above zero
        self.assertGreater(np.min(X), -1.0E-7)

        # Test that it came to equilibrium
        self.assertLess(np.linalg.norm(func(X)) / (1.0 + np.sum(X)), 1E-5)

    def assertConservation(self, y, y0, IDX):
        """Assert the conservation of species throughout the experiment."""
        species_delta = y - y0

        # Check for conservation of species sum
        self.assertAlmostEqual(np.sum(species_delta[IDX]), 0.0, places=5)

    def setUp(self):
        self.ts = np.array([0.0, 100000.0])
        self.y0 = np.random.lognormal(0., 1., 26)
        self.args = np.random.lognormal(0., 1., 14)
        self.tfargs = np.random.lognormal(0., 1., 11)
        # need to convert args from an array to a tuple of numbers

        if (self.tfargs[2] > 1.):
            self.tfargs[2] = self.tfargs[2] - np.floor(self.tfargs[2])

    def test_length(self):
        self.assertEqual(len(dy_dt(self.y0, 0, self.args)), self.y0.size)

    @given(y0=harrays(np.float, 26, elements=floats(0, 10)))
    def test_conservation(self, y0):
        """Check for the conservation of each of the initial receptors."""
        dy = dy_dt(y0, 0.0, self.args)
        #Check for conservation of gc
        self.assertConservation(dy, 0.0, np.array([2, 5, 7, 8, 9, 13, 15, 16, 17, 20, 24, 21, 25]))
        #Check for conservation of IL2Rb
        self.assertConservation(dy, 0.0, np.array([1, 4, 6, 8, 9, 12, 14, 16, 17]))
        #Check for conservation of IL2Ra
        self.assertConservation(dy, 0.0, np.array([0, 3, 6, 7, 9]))
        #Check for conservation of IL15Ra
        self.assertConservation(dy, 0.0, np.array([10, 11, 14, 15, 17]))
        #Check for conservation of IL7Ra
        self.assertConservation(dy, 0.0, np.array([18, 19, 21]))
        #Check for Conservation of IL9R
        self.assertConservation(dy, 0.0, np.array([22, 23, 25]))

    @given(y0=harrays(np.float, 2*26 + 4, elements=floats(0, 10)))
    def test_conservation_full(self, y0):
        """In the absence of trafficking, mass balance should hold in both compartments."""
        kw = np.zeros(11, dtype=np.float64)

        dy = fullModel(y0, 0.0, self.args, kw)

        #Check for conservation of gc
        self.assertConservation(dy, 0.0, np.array([2, 5, 7, 8, 9, 13, 15, 16, 17, 20, 24, 21, 25]))
        #Check for conservation of IL2Rb
        self.assertConservation(dy, 0.0, np.array([1, 4, 6, 8, 9, 12, 14, 16, 17]))
        #Check for conservation of IL2Ra
        self.assertConservation(dy, 0.0, np.array([0, 3, 6, 7, 9]))
        #Check for conservation of IL15Ra
        self.assertConservation(dy, 0.0, np.array([10, 11, 14, 15, 17]))
        #Check for conservation of IL7Ra
        self.assertConservation(dy, 0.0, np.array([18, 19, 21]))
        #Check for Conservation of IL9R
        self.assertConservation(dy, 0.0, np.array([22, 23, 25]))

        #Check for conservation of gc
        self.assertConservation(dy, 0.0, np.array([2, 5, 7, 8, 9, 13, 15, 16, 17, 20, 24, 21, 25]) + 26)
        #Check for conservation of IL2Rb
        self.assertConservation(dy, 0.0, np.array([1, 4, 6, 8, 9, 12, 14, 16, 17]) + 26)
        #Check for conservation of IL2Ra
        self.assertConservation(dy, 0.0, np.array([0, 3, 6, 7, 9]) + 26)
        #Check for conservation of IL15Ra
        self.assertConservation(dy, 0.0, np.array([10, 11, 14, 15, 17]) + 26)
        #Check for conservation of IL7Ra
        self.assertConservation(dy, 0.0, np.array([18, 19, 21]) + 26)
        #Check for Conservation of IL9R
        self.assertConservation(dy, 0.0, np.array([22, 23, 25]) + 26)

    def test_fullModel(self):
        """Assert the two functions solveAutocrine and solveAutocrine complete return the same values."""
        yOut = solveAutocrine(self.tfargs)

        yOut2 = solveAutocrineComplete(self.args, self.tfargs)

        kw = self.args.copy()

        kw[0:4] = 0.

        # Autocrine condition assumes no cytokine present, and so no activity
        self.assertAlmostEqual(getTotalActiveCytokine(0, yOut), 0.0, places=5)

        # Autocrine condition assumes no cytokine present, and so no activity
        self.assertAlmostEqual(getTotalActiveCytokine(0, yOut2), 0.0, places=5)

        self.assertPosEquilibrium(yOut, lambda y: fullModel(y, 0.0, kw, self.tfargs))

    @given(y0=harrays(np.float, 2*26 + 4, elements=floats(0, 10)))
    def test_reproducible(self, y0):

        dy1 = fullModel(y0, 0.0, self.args, self.tfargs)

        dy2 = fullModel(y0, 1.0, self.args, self.tfargs)

        dy3 = fullModel(y0, 2.0, self.args, self.tfargs)

        # Test that there's no difference
        self.assertLess(np.linalg.norm(dy1 - dy2), 1E-8)

        # Test that there's no difference
        self.assertLess(np.linalg.norm(dy1 - dy3), 1E-8)

    #@given(vec=harrays(np.float, 25, elements=floats(0.001, 10.0)), sortF=floats(0.1, 0.9))
    #def test_runCkine(self, vec, sortF):
    #    vec = np.insert(vec, 2, sortF)
    #    # 11 trafRates and 15 rxnRates
    #    trafRates = vec[0:11]
    #    rxnRates = vec[11:26]

    #    ys, retVal = runCkine(self.ts, rxnRates, trafRates)
        
        # test that return value of runCkine isn't negative (model run didn't fail)
    #    self.assertGreaterEqual(retVal, 0)

    def test_jacobian(self):
        '''Compares the approximate Jacobian (approx_jacobian() in Shuffle_ODE.py) with the analytical Jacobian (jacobian() of model.cpp).
        Both Jacobians are evaluating the partial derivatives of dydt.'''
        analytical = jacobian(self.y0, self.ts[0], self.args)
        approx = approx_jac_dydt(self.y0, self.ts[0], self.args, delta=1.0E-4) # Large delta to prevent round-off error  

        self.assertTrue(analytical.shape == approx.shape)

        self.assertTrue(np.allclose(analytical, approx, rtol=0.1, atol=0.1))


    def test_tensor(self):
        tensor = np.random.rand(35,100,20)
        arr = []
        for i in range(1,8):
            R2X = find_R2X(tensor, i)
            arr.append(R2X)
        # confirm R2X for higher components is larger
        for j in range(len(arr)-1):
            self.assertTrue(arr[j] < arr[j+1])
        #confirm R2X is >= 0 and <=1
        self.assertGreaterEqual(np.min(arr),0)
        self.assertLessEqual(np.max(arr),1)


    def test_initial(self):
        #test to check that at least one nonzero is at timepoint zero
        
        t = 60. * 4 # let's let the system run for 4 hours
        ts = np.linspace(0.0, t, 100) #generate 100 evenly spaced timepoints
        
        temp, retVal = runCkine(ts, self.args, self.tfargs)
        self.assertGreater(np.count_nonzero(temp[0,:]), 0)
