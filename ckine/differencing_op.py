"""
Theano Op for using differencing for Jacobian calculation.
"""
import numpy as np
from theano.tensor import dot, dmatrix, dvector, Op


class centralDiff(Op):
    itypes = [dvector]
    otypes = [dvector]

    def __init__(self, calcModel):
        self.M = calcModel
        self.dg = centralDiffGrad(calcModel)

    def perform(self, node, inputs, outputs):
        if np.any(np.greater(inputs[0], 1.0E4)):
            mu = np.full((self.M.concs*2, ), -np.inf, dtype=np.float64)
        else:
            mu = self.M.calc(inputs[0])

        outputs[0][0] = np.array(mu)

    def grad(self, inputs, g):
        """ Calculate the centralDiff gradient. """
        return [dot(g[0], self.dg(inputs[0]))]


class centralDiffGrad(Op):
    itypes = [dvector]
    otypes = [dmatrix]

    def __init__(self, calcModel):
        self.M = calcModel

    def perform(self, node, inputs, outputs):
        # Find our current point
        x0 = inputs[0]

        epsilon = 1.0E-7

        jac = np.full((x0.size, self.M.concs*2), -np.inf, dtype=np.float64)

        if np.all(np.less(inputs[0], 1.0E4)):
            f0 = self.M.calc(x0)

            # Do all the calculations
            for i in range(x0.size):
                dx = x0.copy()
                dx[i] = dx[i] + epsilon
                jac[i] = (self.M.calc(dx) - f0)/epsilon

        outputs[0][0] = np.transpose(jac)
