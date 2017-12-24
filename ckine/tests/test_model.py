import unittest
from ..model import dy_dt, fullModel, solveAutocrine, getTotalActiveCytokine, getActiveSpecies
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays as harrays
import copy


class TestModel(unittest.TestCase):
    def assertPosEquilibrium(self, X, func):
        # All the species abundances should be above zero
        self.assertGreater(np.min(X), -1.0E-7)

        # Test that it came to equilirbium
        self.assertLess(np.linalg.norm(func(X)) / (1.0 + np.sum(X)), 1E-5)

    def assertConservation(self, y, y0, IDX):
        species_delta = y - y0

        # Check for conservation of species sum
        self.assertAlmostEqual(np.sum(species_delta[IDX]), 0.0, places=5)

    def setUp(self):
        self.ts = np.array([0.0, 100000.0])
        self.y0 = np.random.lognormal(0., 1., 26)
        self.args = np.random.lognormal(0., 1., 17)
        self.tfargs = np.random.lognormal(0., 1., 11)
        self.active = getActiveSpecies()
        # need to convert args from an array to a tuple of numbers

    def test_length(self):                        
        self.assertEqual(len(dy_dt(self.y0, 0, self.args)), self.y0.size)
    
    @settings(deadline=None)
    @given(y0=harrays(np.float, 26, elements=floats(0, 10)))
    def test_conservation(self, y0):
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

    @settings(deadline=None)
    @given(y0=harrays(np.float, 2*26 + 4, elements=floats(0, 10)))
    def test_conservation_full(self, y0):
        """In the absence of trafficking, mass balance should hold in both compartments."""
        kw = np.zeros(11, dtype=np.float64)

        dy = fullModel(y0, 0.0, self.args, kw, self.active)
        
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
        yOut = solveAutocrine(self.args, self.tfargs)

        kw = self.args.copy()

        kw[0:4] = 0.

        self.assertPosEquilibrium(yOut, lambda y: fullModel(y, 0.0, kw, self.tfargs, self.active))

        # Autocrine condition assumes no cytokine present, and so no activity
        self.assertAlmostEqual(getTotalActiveCytokine(0, yOut), 0.0, places=5)

    @settings(deadline=None)
    @given(y0=harrays(np.float, 2*26 + 4, elements=floats(0, 10)))
    def test_reproducible(self, y0):

        dy1 = fullModel(y0, 0.0, self.args, self.tfargs, self.active)

        dy2 = fullModel(y0, 1.0, self.args, self.tfargs, self.active)

        dy3 = fullModel(y0, 2.0, self.args, self.tfargs, self.active)

        # Test that there's no difference
        self.assertLess(np.linalg.norm(dy1 - dy2), 1E-8)

        # Test that there's no difference
        self.assertLess(np.linalg.norm(dy1 - dy3), 1E-8)

