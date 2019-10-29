"""
Test that the fitting code can at least build the likelihood model.
"""
import unittest
from ..fit import build_model as build_model2
from ..fit_others import build_model as build_model4


class TestFit(unittest.TestCase):
    """Class to test fitting."""

    def test_fitIL2_15(self):
        """ Test that the IL2/15 model can build. """
        M = build_model2()
        M.build()

    def test_fitIL4_7(self):
        """ Test that the IL4/7 model can build. """
        M = build_model4()
        M.build()
