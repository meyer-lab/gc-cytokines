import unittest
from theano.tests import unittest_tools as utt
import numpy as np
from ..differencing_op import centralDiff
from ..fit import IL2_sum_squared_dist


class TestOp(unittest.TestCase):
    def test_grad(self):
        mdl = IL2_sum_squared_dist()
        mdl.load()

        Opp = centralDiff(mdl, parallel=False)

        XX = np.full(11, 0.5, dtype=np.float64)

        utt.verify_grad(Opp, [XX])

        # Array we've had trouble with before.
        #probb = np.array([9.6680343654, 0.993278390419, 1.17495422785, 0.028015050235, 0.0397793662447, 0.118670968956, 1.9433832377, 0.0407013089368, 10.16551218, 10.88407284, 24.38150978], dtype=np.float64)

        #utt.verify_grad(Opp, [probb])
