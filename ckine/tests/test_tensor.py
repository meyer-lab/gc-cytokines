"""
Unit test file.
"""
import unittest
import numpy as np
import tensorly
from ..Tensor_analysis import find_R2X, perform_decomposition, reorient_factors, scale_all
from ..tensor_generation import findy
tensorly.set_backend('numpy')

class TestModel(unittest.TestCase):
    '''Test Class for Tensor related work.'''
    def test_R2X(self):
        '''Test to ensure R2X for higher components is larger.'''
        tensor = np.random.rand(20,35,100,20)
        arr = []
        for i in range(1,8):
            factors = perform_decomposition(tensor, i)
            R2X = find_R2X(tensor, factors)
            arr.append(R2X)
        for j in range(len(arr)-1):
            self.assertTrue(arr[j] < arr[j+1])
        #confirm R2X is >= 0 and <=1
        self.assertGreaterEqual(np.min(arr),0)
        self.assertLessEqual(np.max(arr),1)

    def test_tensor_parameters(self, r=1):
        '''Function to ensure if rate parameters change in the model code then an error should warn us to update tensor generation code.'''
        y_combos, new_mat = findy(r,50)[0:2]

        self.assertTrue(y_combos.shape[0] == new_mat.shape[0])

    def test_reorientation(self, n_comp = 20):
        """ Test if reorienting the factors matrices changes anything about the original tensor itself. """
        tensor = np.random.rand(20, 35, 100, n_comp)
        factors = perform_decomposition(tensor, n_comp-1)
        reconstruct_old = tensorly.kruskal_to_tensor(factors)
        new_factors = reorient_factors(factors)
        reconstruct_new = tensorly.kruskal_to_tensor(new_factors)
        np.testing.assert_almost_equal(reconstruct_old, reconstruct_new)

    def test_rescale_all(self, n_comp = 20):
        """ Test if rescaling every component keeps the tensor the same. """
        tensor = np.random.rand(20, 35, 100, n_comp)
        factors = perform_decomposition(tensor, n_comp-1)
        reconstruct_old = tensorly.kruskal_to_tensor(factors)
        newfactors = scale_all(factors)
        reconstruct_new = tensorly.kruskal_to_tensor(newfactors)
        np.testing.assert_almost_equal(reconstruct_old, reconstruct_new)
