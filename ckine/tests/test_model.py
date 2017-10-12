import unittest
from ..model import dy_dt
import numpy as np
from scipy.integrate import odeint


class TestModel(unittest.TestCase):
    def setUp(self):
        print("Setup testing")

        self.ts = np.array([0.0, 100000.0])
        self.y0 = np.random.lognormal(0., 1., 10)
        self.args1 = list(np.random.lognormal(0., 1., 7))
        self.args = tuple(self.args1)
        # need to convert args from an array to a tuple of numbers

    def test_length(self):                        
        self.assertEqual(len(dy_dt(self.y0, 0, *self.args)),10)

    def test_equilibrium(self):
        y, fullout = odeint(dy_dt, self.y0, self.ts, self.args,
                    full_output = True, mxstep = 5000)
        z = np.linalg.norm(dy_dt(y[1, :], 0, *self.args)) # have the sum of squares for all the dy_dt values be z
        self.assertTrue(z < 1E-5)
  
    def test_conservation(self):
        y, fullout = odeint(dy_dt, self.y0, self.ts, self.args,
                    full_output = True, mxstep = 5000)
        
        #Assign a value eq for the sum of amounts of receptors at equilirium
        gc_eq = y[1,2] + y[1,5] + y[1,7] + y[1,8] + y[1,9]
        IL2Ra_eq = y[1,0] + y[1,3] + y[1,6] + y[1,7] + y[1,9]
        IL2Rb_eq = y[1,1] + y[1,4] + y[1,6] + y[1,8] + y[1,9]
        
        #Assign a value for the sum of amounts of each receptor at initial conditions
        gc_initial = self.y0[2] + self.y0[5] + self.y0[7] + self.y0[8] + self.y0[9]
        IL2Ra_initial = self.y0[0] + self.y0[3] + self.y0[6] + self.y0[7] + self.y0[9]
        IL2Rb_initial = self.y0[1] + self.y0[4] + self.y0[6] + self.y0[8] + self.y0[9]
        
        #Check for conservation of gc
        self.assertAlmostEqual(gc_eq,gc_initial)

        self.assertAlmostEqual(IL2Rb_eq, IL2Rb_initial)

        self.assertAlmostEqual(IL2Ra_eq, IL2Ra_initial)
