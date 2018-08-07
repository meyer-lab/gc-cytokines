import unittest
import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
import numpy as np
from ..differencing_op import runCkineOp, runCkineKineticOp, runCkineDoseOp, runCkinePreSOp
from ..model import nSpecies, nParams, getTotalActiveSpecies


def setupJacobian(Op, unk):
    a = T.dvector('tempVar')
    fexpr = Op(a)

    # Calculate the Jacobian
    J = theano.scan(lambda i, y, x : T.grad(fexpr[i], a), sequences=T.arange(fexpr.shape[0]), non_sequences=[fexpr, a])[0]

    f = theano.function([a], fexpr)
    fprime = theano.function([a], J)
    
    return f(unk), fprime(unk)


class TestOp(unittest.TestCase):
    def setUp(self):
        self.unkV = np.full(nParams(), 0.3)
        self.doseUnkV = self.unkV[6::]
        self.cond = np.full(nSpecies(), 0.1)
        self.conditions = np.full((3, 6), 10.)
        self.ts = np.linspace(0., 1000.)

    def test_runCkineOp_T0(self):
        utt.verify_grad(runCkineOp(np.array([0.0])), [self.unkV])

    def test_runCkineOp(self):
        utt.verify_grad(runCkineOp(np.array([100.])), [self.unkV])

    @unittest.skip('TODO: Think the sensitivity analysis here is slightly off—this should be able to pass.')
    def test_runCkinePreSOp(self):
        utt.verify_grad(runCkinePreSOp(np.array([100.]), np.array([100.]),
                                       np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])), [self.unkV]) 

    def test_runCkineKineticOp(self):
        utt.verify_grad(runCkineKineticOp(self.ts, self.cond), [self.unkV])

    def test_runCkineDoseOp(self):
        """ Verify the derivative passed back by runCkineDoseOp. """
        Op = runCkineDoseOp(np.array(1.0), self.cond, self.conditions)
        
        utt.verify_grad(Op, [self.doseUnkV])

    def test_runCkineDoseOp_noActivity(self):
        """ Test that in the absence of ligand most values and gradients are zero. """
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
