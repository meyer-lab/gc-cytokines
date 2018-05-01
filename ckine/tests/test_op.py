import unittest
from theano.tests import unittest_tools as utt
import numpy as np
from ..differencing_op import centralDiff, runCkineOp, runCkineKineticOp
from ..fit import IL2Rb_trafficking


class TestOp(unittest.TestCase):
    def test_grad(self):
        mdl = IL2Rb_trafficking()

        XX = np.full(26, 0.5, dtype=np.float64)

        utt.verify_grad(centralDiff(mdl), [XX])

    def test_runCkineOp_T0(self):
        ts = np.array([0.0])

        XX = np.full(26, 0.5, dtype=np.float64)

        utt.verify_grad(runCkineOp(ts), [XX])

    def test_runCkineOp(self):
        ts = np.array([100000.])

        XX = np.full(26, 0.5, dtype=np.float64)

        utt.verify_grad(runCkineOp(ts), [XX], abs_tol=0.01, rel_tol=0.01)

    @unittest.skip("The kinetic FSA Op does not currently work.")
    def test_runCkineKineticOp(self):
        ts = np.linspace(0, 1000, dtype=np.float64)
        cond = np.ones(56, dtype=np.float64)

        XX = np.full(26, 0.1, dtype=np.float64)

        utt.verify_grad(runCkineKineticOp(ts, cond), [XX], abs_tol=0.01, rel_tol=0.01)
