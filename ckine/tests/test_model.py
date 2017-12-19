import unittest
from ..model import dy_dt, fullModel, solveAutocrine, getTotalActiveCytokine
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays as harrays


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
        self.argnames = ('IL2', 'IL15', 'IL7', 'IL9', 'kfwd', 'k5rev', 'k6rev', 'k15rev',
                         'k17rev', 'k18rev', 'k22rev', 'k23rev', 'k26rev', 'k27rev',
                         'k29rev', 'k30rev', 'k31rev')
        self.trafnames = ('endo', 'activeEndo', 'sortF', 'activeSortF', 'kRec', 'kDeg')
        self.args = tuple(list(np.random.lognormal(0., 1., len(self.argnames))))
        self.kwargs = dict(zip(self.argnames, self.args))
        self.endoargs = tuple(list(np.random.lognormal(0., 1., len(self.argnames))))
        self.kwendo = dict(zip(self.trafnames, self.endoargs))
        self.kwendo['exprV'] = np.random.lognormal(0., 1., 6)
        # need to convert args from an array to a tuple of numbers

    def test_length(self):                        
        self.assertEqual(len(dy_dt(self.y0, 0, *self.args)), self.y0.size)
    
    @settings(deadline=None)
    @given(y0=harrays(np.float, 26, elements=floats(0, 10)))
    def test_conservation(self, y0):
        dy = dy_dt(y0, 0.0, *self.args)
        
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

    def test_fullModel(self):
        yOut = solveAutocrine(self.kwargs, self.kwendo)

        self.assertPosEquilibrium(yOut, lambda y: fullModel(y, 0.0, self.kwargs, self.kwendo))

        # Autocrine condition assumes no cytokine present, and so no activity
        self.assertAlmostEqual(getTotalActiveCytokine(0, yOut), 0.0, places=5)
