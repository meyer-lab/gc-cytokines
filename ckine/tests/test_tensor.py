"""
Unit test file.
"""
import unittest
import numpy as np
from hypothesis import settings
from ..Tensor_analysis import find_R2X
from ..tensor_generation import findy

settings.register_profile("ci", max_examples=1000)
#settings.load_profile("ci")

class TestModel(unittest.TestCase):
    '''Test Class for Tensor related work.'''
    def test_R2X(self):
        '''Test to ensure R2X for higher components is larger.'''
        tensor = np.random.rand(35,100,20)
        arr = []
        for i in range(1,8):
            R2X = find_R2X(tensor, i)
            arr.append(R2X)
        for j in range(len(arr)-1):
            self.assertTrue(arr[j] < arr[j+1])
        #confirm R2X is >= 0 and <=1
        self.assertGreaterEqual(np.min(arr),0)
        self.assertLessEqual(np.max(arr),1)

    def test_tensor_parameters(self, r = 1, timelength = 1000):
        '''Function to ensure if rate parameters change in the model code then an error should warn us to update tensor generation code.'''
        y_combos, new_mat = findy(r, timelength)[0:2]
        self.assertEqual(np.array(y_combos.shape).all(), np.array([len(new_mat),timelength, 48]).all())
