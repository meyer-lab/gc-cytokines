"""
Test Theano interfaces and Ops for correctness.
"""
import pytest
import theano
import theano.tensor as T
from tests import unittest_tools as utt
import numpy as np
from ..differencing_op import runCkineDoseOp
from ..model import nSpecies, nParams, rxParams, getTotalActiveSpecies


def setupJacobian(Op, unk):
    """ Take an Op, and a vector at which to evaluate the Op. Pass back the return value and Jacobian. """
    a = T.dvector("tempVar")
    fexpr = Op(a)

    # Calculate the Jacobian
    J = theano.scan(lambda i, y, x: T.grad(fexpr[i], a), sequences=T.arange(fexpr.shape[0]), non_sequences=[fexpr, a])[0]

    f = theano.function([a], fexpr)
    fprime = theano.function([a], J)

    return f(unk), fprime(unk)


@pytest.mark.parametrize("tps", [np.array(1.0), np.logspace(-3, 3, num=10)])
@pytest.mark.parametrize("params", [np.full(nParams() - 6, 0.3), np.full(nParams() - 6, 0.1), np.full(rxParams() - 6, 0.3)])
@pytest.mark.parametrize("preT,preStim", [(0.0, None), (10.0, np.ones(6) * 10.0)])
def test_runCkineDoseOp(tps, params, preT, preStim):
    """ Verify the Jacobian passed back by runCkineDoseOp. """
    theano.config.compute_test_value = "ignore"
    Op = runCkineDoseOp(tps, np.full(nSpecies(), 0.1), np.full((3, 6), 10.0), preT, preStim)

    utt.verify_grad(Op, [params], abs_tol=0.001, rel_tol=0.001)


def test_runCkineDoseOp_noActivity():
    """ Test that in the absence of ligand most values and gradients are zero. """
    theano.config.compute_test_value = "ignore"
    # Setup an Op for conditions with no ligand, looking at cytokine activity
    Op = runCkineDoseOp(np.array(10.0), getTotalActiveSpecies().astype(np.float64), np.zeros((3, 6)))

    # Calculate the Jacobian
    f, Jac = setupJacobian(Op, np.full(nParams() - 6, 0.3))

    # There should be no activity
    np.testing.assert_almost_equal(np.max(f), 0.0)

    # Assert that no other parameters matter when there is no ligand
    np.testing.assert_almost_equal(np.max(np.sum(Jac, axis=0)[6::]), 0.0)

    # Assert that all the conditions are the same so the derivatives are the same
    np.testing.assert_almost_equal(np.std(np.sum(Jac, axis=1)), 0.0)
