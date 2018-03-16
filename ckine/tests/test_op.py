import unittest
from theano.tests import unittest_tools as utt
import numpy as np
from ..differencing_op import centralDiff
from ..fit import IL2_sum_squared_dist


class TestOp(unittest.TestCase):
    def test_grad(self):
        mdl = IL2_sum_squared_dist()

        Opp = centralDiff(mdl)

        XX = np.full(11, 0.5, dtype=np.float64)

        utt.verify_grad(Opp, [XX])
