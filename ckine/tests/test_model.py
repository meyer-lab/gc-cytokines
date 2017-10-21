import unittest
from ..model import dy_dt
import numpy as np
from scipy.integrate import odeint


class TestModel(unittest.TestCase):
    def setUp(self):
        self.ts = np.array([0.0, 100000.0])
        self.y0 = np.random.lognormal(0., 1., 26)
        self.args1 = list(np.random.lognormal(0., 1., 18))
        self.args = tuple(self.args1)
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
        
        #Check for conservation of gc
        self.assertAlmostEqual(np.sum(species_delta[gc_species]), 0.0)
        #Check for conservation of IL2Rb
        self.assertAlmostEqual(np.sum(species_delta[IL2Rb_species]), 0.0)
        #Check for conservation of IL2Ra
        self.assertAlmostEqual(np.sum(species_delta[IL2Ra_species]), 0.0)
        #Check for conservation of IL15Ra
        self.assertAlmostEqual(np.sum(species_delta[IL15Ra_species]), 0.0)
