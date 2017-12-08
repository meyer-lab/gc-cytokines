import unittest
from ..model import dy_dt
import numpy as np
from scipy.integrate import odeint
from ..model import subset_wrapper
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays as harrays


class TestModel(unittest.TestCase):
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

    def test_equilibrium(self):
        y = odeint(dy_dt, self.y0, self.ts, self.args, mxstep = 5000)

        z = np.linalg.norm(dy_dt(y[1, :], 0, *self.args)) # have the sum of squares for all the dy_dt values be z
        self.assertLess(z, 1E-5)
  
    def test_conservation(self):
        y = odeint(dy_dt, self.y0, self.ts, self.args, mxstep = 5000)

        species_delta = y[1, :] - self.y0

        gc_species = np.array([2, 5, 7, 8, 9, 13, 15, 16, 17, 20, 24, 21, 25])
        IL2Ra_species = np.array([0, 3, 6, 7, 9])
        IL2Rb_species = np.array([1, 4, 6, 8, 9, 12, 14, 16, 17])
        IL15Ra_species = np.array([10, 11, 14, 15, 17])
        IL7Ra_species = np.array([18,19,21])
        IL9R_species = np.array([22,23,25])
        
        #Check for conservation of gc
        self.assertAlmostEqual(np.sum(species_delta[gc_species]), 0.0)
        #Check for conservation of IL2Rb
        self.assertAlmostEqual(np.sum(species_delta[IL2Rb_species]), 0.0)
        #Check for conservation of IL2Ra
        self.assertAlmostEqual(np.sum(species_delta[IL2Ra_species]), 0.0)
        #Check for conservation of IL15Ra
        self.assertAlmostEqual(np.sum(species_delta[IL15Ra_species]), 0.0)
        #Check for conservation of IL7Ra
        self.assertAlmostEqual(np.sum(species_delta[IL7Ra_species]), 0.0)
        #Check for Conservation of IL9R
        self.assertAlmostEqual(np.sum(species_delta[IL9R_species]), 0.0)
    
    @settings(deadline=None)
    @given(y0=harrays(np.float, 10, elements=floats(0, 100)), args=harrays(np.float, 4, elements=floats(0.0001, 1)))
    def test_IL2_wrapper(self, y0, args):
        # run odeint on some of the values... make sure they compile correctly and then check the length of the output

        wrap = lambda y, t: subset_wrapper(y, t, IL2i=args[0], kfwd=args[1], k5rev=args[2], k6rev=args[3])

        retval = odeint(wrap, y0, self.ts, mxstep=9000)

        # Diff the starting and stopping state to check mass conservation
        species_delta = retval[1, :] - y0

        self.assertEqual(len(retval[1]), 10)

        # Check for conservation of gc
        self.assertAlmostEqual(np.sum(species_delta[np.array([2, 5, 7, 8, 9])]), 0.0, places=5)
        # Check for conservation of IL2Ra
        self.assertAlmostEqual(np.sum(species_delta[np.array([0, 3, 6, 7, 9])]), 0.0, places=5)
        # Check for conservation of IL2Rb
        self.assertAlmostEqual(np.sum(species_delta[np.array([1, 4, 6, 8, 9])]), 0.0, places=5)
        # TODO: Add mass balance checks here.
    
    @settings(deadline=None)
    @given(y0=harrays(np.float, 10, elements=floats(0, 100)), args=harrays(np.float, 6, elements=floats(0.0001, 1)))
    def test_IL15_wrapper(self, y0, args):
        wrap = lambda y, t: subset_wrapper(y, t, IL15i=1.0, kfwd=args[0], k15rev=args[1], k17rev=args[2], k18rev=args[3], k22rev=args[4], k23rev=args[5])

        retval = odeint(wrap, y0, self.ts, mxstep = 6000)
        self.assertEqual(len(retval[1]), 10)
        # TODO: Add mass balance checks here.
