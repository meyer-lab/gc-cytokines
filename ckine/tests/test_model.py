"""
Unit test file.
"""
import unittest
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays as harrays
from ..model import dy_dt, fullModel, solveAutocrine, getTotalActiveCytokine, solveAutocrineComplete, runCkine, runCkineU, jacobian, fullJacobian
from ..util_analysis.Shuffle_ODE import approx_jacobian
from ..Tensor_analysis import find_R2X

settings.register_profile("ci", max_examples=1000)
#settings.load_profile("ci")

conservation_IDX = [np.array([1, 4, 5, 7, 8, 11, 12, 14, 15]), # IL2Rb
                    np.array([0, 3, 5, 6, 8]), # IL2Ra
                    np.array([9, 10, 12, 13, 15]), # IL15Ra
                    np.array([16, 17, 18]), # IL7Ra
                    np.array([19, 20, 21]), # IL9R
                    np.array([2, 6, 7, 8, 13, 14, 15, 18, 21])] # gc

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
        self.assertAlmostEqual(np.sum(species_delta[IDX]), 0.0, msg=str(IDX))

    def setUp(self):
        self.ts = np.array([0.0, 1000.0])
        self.y0 = np.random.lognormal(0., 1., 22)
        self.args = np.random.lognormal(0., 1., 13)
        self.tfargs = np.random.lognormal(0., 1., 11)
        self.fully = np.random.lognormal(0., 1., 48)

        # Force sorting fraction to be less than 1.0
        self.tfargs[2] = np.tanh(self.tfargs[2])*0.9

        self.rxntfR = np.concatenate((self.args, self.tfargs))

    def test_length(self):
        self.assertEqual(len(dy_dt(self.y0, 0, self.args)), self.y0.size)

    @given(y0=harrays(np.float, 22, elements=floats(1, 10)))
    def test_conservation(self, y0):
        """Check for the conservation of each of the initial receptors."""
        dy = dy_dt(y0, 0.0, self.args)
        # Check for conservation of each receptor
        for idxs in conservation_IDX:
            self.assertConservation(dy, 0.0, idxs)

    @given(y0=harrays(np.float, 2*22 + 4, elements=floats(1, 10)))
    def test_conservation_full(self, y0):
        """In the absence of trafficking, mass balance should hold in both compartments."""
        kw = np.zeros(self.tfargs.shape, dtype=np.float64)

        dy = fullModel(y0, 0.0, self.args, kw)

        # Check for conservation of each surface receptor
        for idxs in conservation_IDX:
            self.assertConservation(dy, 0.0, idxs)

        # Check for conservation of each endosomal receptor
        for idxs in conservation_IDX:
            self.assertConservation(dy, 0.0, idxs + 22)

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

    @given(y0=harrays(np.float, 2*22 + 4, elements=floats(0, 10)))
    def test_reproducible(self, y0):

        dy1 = fullModel(y0, 0.0, self.args, self.tfargs)

        dy2 = fullModel(y0, 1.0, self.args, self.tfargs)

        dy3 = fullModel(y0, 2.0, self.args, self.tfargs)

        # Test that there's no difference
        self.assertLess(np.linalg.norm(dy1 - dy2), 1E-8)

        # Test that there's no difference
        self.assertLess(np.linalg.norm(dy1 - dy3), 1E-8)

    @given(vec=harrays(np.float, 24, elements=floats(0.1, 10.0)))
    def test_runCkine(self, vec):
        # Force sorting fraction to be less than 1.0
        vec[15] = np.tanh(vec[15])*0.9

        retVal = runCkineU(self.ts, vec)[1]

        # test that return value of runCkine isn't negative (model run didn't fail)
        self.assertGreaterEqual(retVal, 0)

    def test_jacobian(self):
        '''Compares the approximate Jacobian (approx_jacobian() in Shuffle_ODE.py) with the analytical Jacobian (jacobian() of model.cpp).
        Both Jacobians are evaluating the partial derivatives of dydt.'''
        analytical = jacobian(self.y0, self.ts[0], self.args)
        approx = approx_jacobian(lambda x: dy_dt(x, self.ts[0], self.args), self.y0, delta=1.0E-4) # Large delta to prevent round-off error

        closeness = np.isclose(analytical, approx, rtol=0.0001, atol=0.0001)

        if not np.all(closeness):
            IDXdiff = np.where(np.logical_not(closeness))
            print(IDXdiff)
            print(analytical[IDXdiff])
            print(approx[IDXdiff])

        self.assertTrue(np.all(closeness))

    def test_fullJacobian(self):
        analytical = fullJacobian(self.fully, 0.0, np.concatenate((self.args, self.tfargs)))
        approx = approx_jacobian(lambda x: fullModel(x, 0.0, self.args, self.tfargs), self.fully, delta = 1.0E-6)

        self.assertTrue(analytical.shape == approx.shape)

        closeness = np.isclose(analytical, approx, rtol=0.0001, atol=0.0001)

        if not np.all(closeness):
            IDXdiff = np.where(np.logical_not(closeness))
            print(IDXdiff)
            print(analytical[IDXdiff])
            print(approx[IDXdiff])

        self.assertTrue(np.all(closeness))

    def test_tensor(self):
        tensor = np.random.rand(35, 100, 20)
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
        temp, retVal = runCkine(self.ts, self.args, self.tfargs)
        self.assertGreater(np.count_nonzero(temp[0,:]), 0)
        self.assertGreaterEqual(retVal, 0)

    def test_gc(self):
        ''' Test to check that no active species is present when gamma chain is not expressed. '''
        rxntfR = self.rxntfR.copy()
        rxntfR[20] = 0.0 # set expression of gc to 0.0
        yOut, retVal = runCkineU(self.ts, rxntfR)
        self.assertGreaterEqual(retVal, 0)
        self.assertAlmostEqual(getTotalActiveCytokine(0, yOut[1]), 0.0, places=5) # IL2
        self.assertAlmostEqual(getTotalActiveCytokine(1, yOut[1]), 0.0, places=5) # IL15
        self.assertAlmostEqual(getTotalActiveCytokine(2, yOut[1]), 0.0, places=5) # IL7
        self.assertAlmostEqual(getTotalActiveCytokine(3, yOut[1]), 0.0, places=5) # IL9

    def test_endosomalCTK_bound(self):
        ''' Test that appreciable cytokine winds up in the endosome. '''
        rxntfR = self.rxntfR.copy()
        rxntfR[4:24] = 0.1 # Fill all in to avoid parameter variation
        rxntfR[4] = 0.0001 # Damp down kfwd
        rxntfR[14] = 10.0 # Turn up active endocytosis
        rxntfR[17] = 0.02 # Turn down degradation
        rxntfR[18:24] = 1.0 # Control expression

        rxntfR[0:4] = 0.0
        # set high concentration of IL2
        rxntfR_1 = rxntfR.copy()
        rxntfR_1[0] = 1000.
        # set high concentration of IL15
        rxntfR_2 = rxntfR.copy()
        rxntfR_2[1] = 1000.
        # set high concentration of IL7
        rxntfR_3 = rxntfR.copy()
        rxntfR_3[2] = 1000.
        # set high concentration of IL9
        rxntfR_4 = rxntfR.copy()
        rxntfR_4[3] = 1000.

        # first element is t=0 and second element is t=10**5
        yOut_1, retVal = runCkineU(self.ts, rxntfR_1)
        self.assertGreaterEqual(retVal, 0)
        yOut_2, retVal = runCkineU(self.ts, rxntfR_2)
        self.assertGreaterEqual(retVal, 0)
        yOut_3, retVal = runCkineU(self.ts, rxntfR_3)
        self.assertGreaterEqual(retVal, 0)
        yOut_4, retVal = runCkineU(self.ts, rxntfR_4)
        self.assertGreaterEqual(retVal, 0)

        # make sure endosomal free ligand is positive at equilibrium
        # IL2
        self.assertGreater(yOut_1[1, 44], 0)
        self.assertTrue((yOut_1[1, 45:48] == 0).all()) # no other ligand
        # IL15
        self.assertGreater(yOut_2[1, 45], 0)
        self.assertTrue(yOut_2[1, 44] == 0) # no other ligand
        self.assertTrue((yOut_2[1, 46:48] == 0).all()) # no other ligand
        # IL7
        self.assertGreater(yOut_3[1, 46], 0)
        self.assertTrue((yOut_3[1, 44:46] == 0).all()) # no other ligand
        self.assertTrue(yOut_3[1, 47] == 0) # no other ligand
        # IL9
        self.assertGreater(yOut_4[1, 47], 0)
        self.assertTrue((yOut_4[1, 44:47] == 0).all()) # no other ligand

        # make sure total amount of ligand bound to receptors is positive at equilibrium
        self.assertTrue(np.greater(yOut_1[25:31], 0.0).all())
        self.assertTrue(np.greater(yOut_2[32:38], 0.0).all())
        self.assertTrue(np.greater(yOut_3[39:41], 0.0).all())
        self.assertTrue(np.greater(yOut_4[42:44], 0.0).all())
