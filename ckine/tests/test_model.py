import unittest
from ..model import dy_dt
import numpy as np
from scipy.integrate import odeint
from ..model import subset_wrapper
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays as harrays


class TestModel(unittest.TestCase):
    def assertPosEquilibrium(self, X, func):
        # All the species abundances should be above zero
        self.assertGreater(np.min(X), -1.0E-8)

        # Test that it came to equilirbium
        self.assertLess(np.linalg.norm(func(X, 0)) / (1.0 + np.sum(X)), 2E-6)

    def assertConservation(self, y, y0, IDX):
        species_delta = y - y0

        # Check for conservation of species sum
        self.assertAlmostEqual(np.sum(species_delta[IDX]), 0.0, places=5)

    def setUp(self):
        self.ts = np.array([0.0, 100000.0])
        self.y0 = np.random.lognormal(0., 1., 26)
        self.args1 = list(np.random.lognormal(0., 1., 17))
        self.args = tuple(self.args1)

        self.argnames = ('IL2', 'IL15', 'IL7', 'IL9', 'kfwd', 'k5rev', 'k6rev', 'k15rev',
                         'k17rev', 'k18rev', 'k22rev', 'k23rev', 'k26rev', 'k27rev',
                         'k29rev', 'k30rev', 'k31rev')
        self.kwargs = dict(zip(self.argnames, self.args1))
        # need to convert args from an array to a tuple of numbers

    def test_length(self):                        
        self.assertEqual(len(dy_dt(self.y0, 0, *self.args)), self.y0.size)
  
    def test_conservation(self):
        y = odeint(dy_dt, self.y0, self.ts, self.args, mxstep = 5000)
        
        #Check for conservation of gc
        self.assertConservation(y[1, :], self.y0, np.array([2, 5, 7, 8, 9, 13, 15, 16, 17, 20, 24, 21, 25]))
        #Check for conservation of IL2Rb
        self.assertConservation(y[1, :], self.y0, np.array([1, 4, 6, 8, 9, 12, 14, 16, 17]))
        #Check for conservation of IL2Ra
        self.assertConservation(y[1, :], self.y0, np.array([0, 3, 6, 7, 9]))
        #Check for conservation of IL15Ra
        self.assertConservation(y[1, :], self.y0, np.array([10, 11, 14, 15, 17]))
        #Check for conservation of IL7Ra
        self.assertConservation(y[1, :], self.y0, np.array([18, 19, 21]))
        #Check for Conservation of IL9R
        self.assertConservation(y[1, :], self.y0, np.array([22, 23, 25]))

        # Assert positive and at equilibrium
        self.assertPosEquilibrium(y[1, :], lambda y, t: dy_dt(y, t, *self.args))
    
    @settings(deadline=None)
    @given(y0=harrays(np.float, 10, elements=floats(0, 100)), args=harrays(np.float, 4, elements=floats(0.0001, 1)))
    def test_IL2_wrapper(self, y0, args):
        # run odeint on some of the values... make sure they compile correctly and then check the length of the output

        wrap = lambda y, t: subset_wrapper(y, t, IL2i=args[0], kfwd=args[1], k5rev=args[2], k6rev=args[3])

        retval = odeint(wrap, y0, self.ts, mxstep=9000)

        self.assertEqual(len(retval[1]), 10)

        # Check for conservation of gc
        self.assertConservation(retval[1, :], y0, np.array([2, 5, 7, 8, 9]))
        # Check for conservation of IL2Ra
        self.assertConservation(retval[1, :], y0, np.array([0, 3, 6, 7, 9]))
        # Check for conservation of IL2Rb
        self.assertConservation(retval[1, :], y0, np.array([1, 4, 6, 8, 9]))

        # Assert positive and at equilibrium
        self.assertPosEquilibrium(retval[1, :], wrap)
    
    @settings(deadline=None)
    @given(y0=harrays(np.float, 10, elements=floats(0, 100)), args=harrays(np.float, 6, elements=floats(0.0001, 1)))
    def test_IL15_wrapper(self, y0, args):
        wrap = lambda y, t: subset_wrapper(y, t, IL15i=1.0, kfwd=args[0], k15rev=args[1], k17rev=args[2], k18rev=args[3], k22rev=args[4], k23rev=args[5])

        retval = odeint(wrap, y0, self.ts, mxstep = 6000)
        self.assertEqual(len(retval[1]), 10)
        # TODO: Add mass balance checks here.


        # Assert positive and at equilibrium
        self.assertPosEquilibrium(retval[1, :], wrap)
