"""
Unit test file.
"""
import unittest
import numpy as np
import tensorly as tl
from ..Tensor_analysis import find_R2X, perform_decomposition, scale_all
from ..tensor_generation import findy

class TestModel(unittest.TestCase):
    '''Test Class for Tensor related work.'''
    def test_R2X(self):
        '''Test to ensure R2X for higher components is larger.'''
        tensor = tl.tensor(np.random.rand(12, 10, 15))
        arr = []
        for i in range(1, 8):
            factors = perform_decomposition(tensor, i)
            R2X = find_R2X(tensor, factors)
            arr.append(R2X)
        for j in range(len(arr)-1):
            self.assertTrue(arr[j] < arr[j+1])
        #confirm R2X is >= 0 and <=1
        self.assertGreaterEqual(tl.min(arr), 0)
        self.assertLessEqual(tl.max(arr), 1)

    def test_rescale_all(self, n_comp = 20):
        """Test if rescaling every component keeps the tensor the same."""
        tensor = tl.tensor(np.random.rand(20, 35, n_comp))
        factors = perform_decomposition(tensor, n_comp-1)
        reconstruct_old = tl.kruskal_to_tensor(factors)
        newfactors = scale_all(factors)
        reconstruct_new = tl.kruskal_to_tensor(newfactors)
        np.testing.assert_almost_equal(tl.to_numpy(reconstruct_old), tl.to_numpy(reconstruct_new))
