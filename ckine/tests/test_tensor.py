"""
Unit test file.
"""
import unittest
import numpy as np
import tensorly as tl
from ..tensor import find_R2X, perform_decomposition


class TestModel(unittest.TestCase):
    """Test Class for Tensor related work."""

    def test_R2X(self):
        """Test to ensure R2X for higher components is larger."""
        tensor = tl.tensor(np.random.rand(12, 10, 15))
        arr = [find_R2X(tensor, perform_decomposition(tensor, i)) for i in range(1, 8)]

        for j in range(len(arr) - 1):
            self.assertTrue(arr[j] < arr[j + 1])

        # confirm R2X is >= 0 and <=1
        self.assertGreaterEqual(tl.min(arr), 0)
        self.assertLessEqual(tl.max(arr), 1)
