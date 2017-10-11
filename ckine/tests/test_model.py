import unittest
from ..model import dy_dt
import numpy as np
from scipy.integrate import odeint


class TestModel(unittest.TestCase):
    def setUp(self):
        print("Setup testing")
        
    def test_equilibrium(self):
        ts = np.array([0.0, 100000.0])
        y0 = np.ones((10, ), dtype = np.float64)
        args = (1., 1., 1., 1., 1., 1., 0.5) # these 7 float values represent the inputs IL2 through k11rev
        y, fullout = odeint(dy_dt, y0, ts, args,
                    full_output = True, mxstep = 5000)
        z = np.linalg.norm(dy_dt(y[1, :], 0, *args)) # have the sum of squares for all the dy_dt values be z
        self.assertTrue(z < 1E-5)
        
    def test_conservation(self):
        ts = np.array([0.0, 100000.0])
        y0 = np.ones((10, ), dtype = np.float64)
        args = (1., 1., 1., 1., 1., 1., 0.5) # these 7 float values represent the inputs IL2 through k11rev
        y, fullout = odeint(dy_dt, y0, ts, args,
                    full_output = True, mxstep = 5000)
        
       #Assign a value eq for the sum of amounts of receptors at equilirium
        gc_eq = y[1,2] + y[1,5] + y[1,7] + y[1,8] + y[1,9]
        IL2Ra_eq = y[1,0] + y[1,3] + y[1,6] + y[1,7] + y[1,9]
        IL2Rb_eq = y[1,1] + y[1,4] + y[1,6] + y[1,8] + y[1,9]
        
        #Assign a value for the sum of amounts of each receptor at initial conditions
        gc_initial = y0[2] + y0[5] + y0[7] + y0[8] + y0[9]
        IL2Ra_initial = y0[0] + y0[3] + y0[6] + y0[7] + y0[9]
        IL2Rb_initial = y0[1] + y0[4] + y0[6] + y0[8] + y0[9]
        
        #Check for conservation of gc
        self.assertAlmostEqual(gc_eq,gc_initial)

        self.assertAlmostEqual(IL2Rb_eq, IL2Rb_initial)

        self.assertAlmostEqual(IL2Ra_eq, IL2Ra_initial)
