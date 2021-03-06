"""
Unit test file.
"""
import unittest
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays as harrays
from ..model import fullModel, getTotalActiveCytokine, nSpecies, runCkineUP


settings.register_profile("ci", max_examples=1000, deadline=None)
settings.load_profile("ci")

conservation_IDX = [
    np.array([1, 4, 5, 7, 8, 11, 12, 14, 15]),  # IL2Rb
    np.array([0, 3, 5, 6, 8]),  # IL2Ra
    np.array([9, 10, 12, 13, 15]),  # IL15Ra
    np.array([16, 17, 18]),  # IL7Ra
    np.array([19, 20, 21]),  # IL9R
    np.array([22, 23, 24]),  # IL4Ra
    np.array([25, 26, 27]),  # IL21Ra
    np.array([2, 6, 7, 8, 13, 14, 15, 18, 21, 24, 27]),
]  # gc


class TestModel(unittest.TestCase):
    """ Here are the unit tests. """

    def assertPosEquilibrium(self, X, func):
        """Assert that all species came to equilibrium."""
        # All the species abundances should be above zero
        self.assertGreater(np.min(X), -1.0e-7)

        # Test that it came to equilibrium
        self.assertLess(np.linalg.norm(func(X)) / (1.0 + np.sum(X)), 1e-5)

    def assertConservation(self, y, y0, IDX):
        """Assert the conservation of species throughout the experiment."""
        species_delta = y - y0

        # Check for conservation of species sum
        self.assertAlmostEqual(np.sum(species_delta[IDX]), 0.0, msg=str(IDX))

    def setUp(self):
        self.ts = np.array([0.0, 100000.0])
        self.y0 = np.random.lognormal(0.0, 1.0, 28)
        self.args = np.random.lognormal(0.0, 1.0, 17)
        self.tfargs = np.random.lognormal(0.0, 1.0, 13)
        self.fully = np.random.lognormal(0.0, 1.0, 62)

        # Force sorting fraction to be less than 1.0
        self.tfargs[2] = np.tanh(self.tfargs[2]) * 0.9

        self.rxntfR = np.concatenate((self.args, self.tfargs))

    @given(y0=harrays(np.float, nSpecies(), elements=floats(1, 10)))
    def test_conservation_full(self, y0):
        """In the absence of trafficking, mass balance should hold in both compartments."""
        rxntfR = self.rxntfR.copy()
        rxntfR[17:30] = 0.0

        dy = fullModel(y0, 0.0, rxntfR)

        # Check for conservation of each surface receptor
        for idxs in conservation_IDX:
            self.assertConservation(dy, 0.0, idxs)

        # Check for conservation of each endosomal receptor
        for idxs in conservation_IDX:
            self.assertConservation(dy, 0.0, idxs + 28)

    def test_equlibrium(self):
        """System should still come to equilibrium after being stimulated with ligand"""
        t = np.array([0.0, 100000.0])
        rxn = self.rxntfR.copy()
        rxn[0:6] = 0.0  # set ligands to 0
        rxnIL2, rxnIL15, rxnIL7, rxnIL9, rxnIL4, rxnIL21 = rxn.copy(), rxn.copy(), rxn.copy(), rxn.copy(), rxn.copy(), rxn.copy()
        rxnIL2[0], rxnIL15[1], rxnIL7[2], rxnIL9[3], rxnIL4[5], rxnIL21[6] = 100.0, 100.0, 100.0, 100.0, 100.0, 100.0

        # runCkine to get yOut
        yOut_2 = runCkineUP(t, rxnIL2)
        yOut_15 = runCkineUP(t, rxnIL15)
        yOut_7 = runCkineUP(t, rxnIL7)
        yOut_9 = runCkineUP(t, rxnIL9)
        yOut_4 = runCkineUP(t, rxnIL4)
        yOut_21 = runCkineUP(t, rxnIL21)

        # check that dydt is ~0
        self.assertPosEquilibrium(yOut_2[1], lambda y: fullModel(y, t[1], rxnIL2))
        self.assertPosEquilibrium(yOut_15[1], lambda y: fullModel(y, t[1], rxnIL15))
        self.assertPosEquilibrium(yOut_7[1], lambda y: fullModel(y, t[1], rxnIL7))
        self.assertPosEquilibrium(yOut_9[1], lambda y: fullModel(y, t[1], rxnIL9))
        self.assertPosEquilibrium(yOut_4[1], lambda y: fullModel(y, t[1], rxnIL4))
        self.assertPosEquilibrium(yOut_21[1], lambda y: fullModel(y, t[1], rxnIL21))

    def test_fullModel(self):
        """ Assert that we're at autocrine steady-state at t=0. """
        yOut = runCkineUP(np.array([0.0]), self.rxntfR)
        yOut = np.squeeze(yOut)

        rxnNoLigand = self.rxntfR
        rxnNoLigand[0:6] = 0.0

        # Autocrine condition assumes no cytokine present, and so no activity
        self.assertAlmostEqual(getTotalActiveCytokine(0, yOut), 0.0, places=5)

        self.assertPosEquilibrium(yOut, lambda y: fullModel(y, 0.0, rxnNoLigand))

    @given(y0=harrays(np.float, nSpecies(), elements=floats(0, 10)))
    def test_reproducible(self, y0):
        """ Make sure full model is reproducible under same conditions. """

        dy1 = fullModel(y0, 0.0, self.rxntfR)

        # Test that there's no difference
        self.assertLess(np.linalg.norm(dy1 - fullModel(y0, 1.0, self.rxntfR)), 1e-8)

        # Test that there's no difference
        self.assertLess(np.linalg.norm(dy1 - fullModel(y0, 2.0, self.rxntfR)), 1e-8)

    @given(vec=harrays(np.float, 30, elements=floats(0.1, 10.0)))
    def test_runCkine(self, vec):
        """ Make sure model runs properly by checking retVal. """
        vec[19] = np.tanh(vec[19]) * 0.9  # Force sorting fraction to be less than 1.0
        runCkineUP(self.ts, vec)

    def test_runCkineParallel(self):
        """ Test that we can run solving in parallel. """
        rxntfr = np.reshape(np.tile(self.rxntfR, 20), (20, -1))

        outt = runCkineUP(self.ts[1], rxntfr)

        # test that all of the solutions returned are identical
        for ii in range(rxntfr.shape[0]):
            self.assertTrue(np.all(outt[0, :] == outt[ii, :]))

    def test_initial(self):
        """ Test that there is at least 1 non-zero species at T=0. """
        temp = runCkineUP(self.ts, self.rxntfR)
        self.assertGreater(np.count_nonzero(temp[0, :]), 0)

    def test_gc(self):
        """ Test to check that no active species is present when gamma chain is not expressed. """
        rxntfR = self.rxntfR.copy()
        rxntfR[24] = 0.0  # set expression of gc to 0.0
        yOut = runCkineUP(self.ts, rxntfR)
        self.assertAlmostEqual(getTotalActiveCytokine(0, yOut[1]), 0.0, places=5)  # IL2
        self.assertAlmostEqual(getTotalActiveCytokine(1, yOut[1]), 0.0, places=5)  # IL15
        self.assertAlmostEqual(getTotalActiveCytokine(2, yOut[1]), 0.0, places=5)  # IL7
        self.assertAlmostEqual(getTotalActiveCytokine(3, yOut[1]), 0.0, places=5)  # IL9
        self.assertAlmostEqual(getTotalActiveCytokine(4, yOut[1]), 0.0, places=5)  # IL4
        self.assertAlmostEqual(getTotalActiveCytokine(5, yOut[1]), 0.0, places=5)  # IL21

    def test_endosomalCTK_bound(self):
        """ Test that appreciable cytokine winds up in the endosome. """
        rxntfR = self.rxntfR.copy()
        rxntfR[0:6] = 0.0
        rxntfR[6] = 1.0e-6  # Damp down kfwd
        rxntfR[7:22] = 0.1  # Fill all in to avoid parameter variation
        rxntfR[18] = 10.0  # Turn up active endocytosis
        rxntfR[21] = 0.02  # Turn down degradation
        rxntfR[22:30] = 10.0  # Control expression

        # set high concentration of IL2
        rxntfR_1 = rxntfR.copy()
        rxntfR_1[0] = 1000.0
        # set high concentration of IL15
        rxntfR_2 = rxntfR.copy()
        rxntfR_2[1] = 1000.0
        # set high concentration of IL7
        rxntfR_3 = rxntfR.copy()
        rxntfR_3[2] = 1000.0
        # set high concentration of IL9
        rxntfR_4 = rxntfR.copy()
        rxntfR_4[3] = 1000.0
        # set high concentration of IL4
        rxntfR_5 = rxntfR.copy()
        rxntfR_5[4] = 1000.0
        # set high concentration of IL21
        rxntfR_6 = rxntfR.copy()
        rxntfR_6[5] = 1000.0

        # first element is t=0 and second element is t=10**5
        yOut_1 = runCkineUP(self.ts, rxntfR_1)
        yOut_2 = runCkineUP(self.ts, rxntfR_2)
        yOut_3 = runCkineUP(self.ts, rxntfR_3)
        yOut_4 = runCkineUP(self.ts, rxntfR_4)
        yOut_5 = runCkineUP(self.ts, rxntfR_5)
        yOut_6 = runCkineUP(self.ts, rxntfR_6)

        # make sure endosomal free ligand is positive at equilibrium
        # IL2
        self.assertGreater(yOut_1[1, 56], 1.0)
        self.assertLess(np.sum(yOut_1[1, np.array([57, 58, 59, 60, 61])]), 1.0e-9)  # no other ligand
        # IL15
        self.assertGreater(yOut_2[1, 57], 1.0)
        self.assertLess(np.sum(yOut_2[1, np.array([56, 58, 59, 60, 61])]), 1.0e-9)  # no other ligand
        # IL7
        self.assertGreater(yOut_3[1, 58], 1.0)
        self.assertLess(np.sum(yOut_3[1, np.array([56, 57, 59, 60, 61])]), 1.0e-9)  # no other ligand
        # IL9
        self.assertGreater(yOut_4[1, 59], 1.0)
        self.assertLess(np.sum(yOut_4[1, np.array([56, 57, 58, 60, 61])]), 1.0e-9)  # no other ligand
        # IL4
        self.assertGreater(yOut_5[1, 60], 1.0)
        self.assertLess(np.sum(yOut_5[1, np.array([56, 57, 58, 59, 61])]), 1.0e-9)  # no other ligand
        # IL21
        self.assertGreater(yOut_6[1, 61], 1.0)
        self.assertLess(np.sum(yOut_6[1, np.array([56, 57, 58, 59, 60])]), 1.0e-9)  # no other ligand

        # make sure total amount of ligand bound to receptors is positive at equilibrium
        self.assertTrue(np.greater(yOut_1[31:37], 0.0).all())
        self.assertTrue(np.greater(yOut_2[38:44], 0.0).all())
        self.assertTrue(np.greater(yOut_3[45:47], 0.0).all())
        self.assertTrue(np.greater(yOut_4[48:50], 0.0).all())
        self.assertTrue(np.greater(yOut_5[51:53], 0.0).all())
        self.assertTrue(np.greater(yOut_6[54:56], 0.0).all())

    def test_noTraff(self):
        """ Make sure no endosomal species are found when endo=0. """
        rxntfR = self.rxntfR.copy()
        rxntfR[17:19] = 0.0  # set endo and activeEndo to 0.0
        yOut = runCkineUP(self.ts, rxntfR)
        tot_endo = np.sum(yOut[1, 28::])
        self.assertEqual(tot_endo, 0.0)
