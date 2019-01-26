"""
Test Theano interfaces and Ops for correctness.
"""
import unittest
import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
import numpy as np
from ..differencing_op import runCkineDoseOp
from ..model import nSpecies, nParams, getTotalActiveSpecies


def setupJacobian(Op, unk):
    """ Take an Op, and a vector at which to evaluate the Op. Pass back the return value and Jacobian. """
    a = T.dvector('tempVar')
    fexpr = Op(a)

    # Calculate the Jacobian
    J = theano.scan(lambda i, y, x : T.grad(fexpr[i], a), sequences=T.arange(fexpr.shape[0]), non_sequences=[fexpr, a])[0]

    f = theano.function([a], fexpr)
    fprime = theano.function([a], J)

    return f(unk), fprime(unk)


class TestOp(unittest.TestCase):
    """ Test Theano Ops. """
    def setUp(self):
        self.unkV = np.full(nParams(), 0.3)
        self.doseUnkV = self.unkV[6::]
        self.cond = np.full(nSpecies(), 0.1)
        self.conditions = np.full((3, 6), 10.)
        self.ts = np.logspace(-3, 3, num=10)

    def test_runCkineDoseOp(self):
        """ Verify the Jacobian passed back by runCkineDoseOp. """
        theano.config.compute_test_value = 'ignore'
        Op = runCkineDoseOp(np.array(1.0), self.cond, self.conditions)

        utt.verify_grad(Op, [self.doseUnkV], abs_tol=0.01, rel_tol=0.01)

    def test_runCkineDoseTpsOp(self):
        """ Verify the Jacobian passed back by runCkineDoseOp. """
        theano.config.compute_test_value = 'ignore'
        Op = runCkineDoseOp(self.ts, self.cond, self.conditions)

        utt.verify_grad(Op, [self.doseUnkV], abs_tol=0.01, rel_tol=0.01)

    def test_runCkineDosePrestimOp(self):
        """ Verify the Jacobian passed back by runCkineDoseOp with prestimulation. """
        Op = runCkineDoseOp(np.array(1.0), self.cond, self.conditions, 10.0, np.ones(6)*10.0)

        utt.verify_grad(Op, [self.doseUnkV], abs_tol=0.01, rel_tol=0.01)

    def test_runCkineDoseOp_noActivity(self):
        """ Test that in the absence of ligand most values and gradients are zero. """
        theano.config.compute_test_value = 'ignore'
        # Setup an Op for conditions with no ligand, looking at cytokine activity
        Op = runCkineDoseOp(np.array(10.0), getTotalActiveSpecies().astype(np.float64), np.zeros_like(self.conditions))

        # Calculate the Jacobian
        f, Jac = setupJacobian(Op, self.doseUnkV)

        # There should be no activity
        self.assertAlmostEqual(np.max(f), 0.0)

        # Assert that no other parameters matter when there is no ligand
        self.assertAlmostEqual(np.max(np.sum(Jac, axis=0)[6::]), 0.0)

        # Assert that all the conditions are the same so the derivatives are the same
        self.assertAlmostEqual(np.std(np.sum(Jac, axis=1)), 0.0)
