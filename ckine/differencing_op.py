"""
Theano Op for using differencing for Jacobian calculation.
"""
import numpy as np
from theano.tensor import dot, dmatrix, dvector, Op
from .model import runCkineU, runCkineSensi


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


class runCkineOp(Op):
    itypes = [dvector]
    otypes = [dvector]

    def __init__(self, ts):
        if ts.size > 1:
            raise NotImplementedError('This Op only works with a single time point.')

        self.ts = ts

    def infer_shape(self, node, i0_shapes):
        assert len(i0_shapes) == 1
        return [(56, )]

    def perform(self, node, inputs, outputs):
        yOut, retVal = runCkineU(self.ts, inputs[0])

        assert yOut.size == 56

        if retVal < 0:
            yOut[:] = -np.inf

        outputs[0][0] = np.squeeze(yOut)

    def grad(self, inputs, g):
        """ Calculate the runCkineOp gradient. """
        return [dot(g[0], runCkineOpDiff(self.ts)(inputs[0]))]


class runCkineOpDiff(Op):
    itypes = [dvector]
    otypes = [dmatrix]

    def __init__(self, ts):
        if ts.size > 1:
            raise NotImplementedError('This Op only works with a single time point.')

        self.ts = ts

    def perform(self, node, inputs, outputs):
        _, retVal, sensi = runCkineSensi(self.ts, inputs[0])

        if retVal < 0:
            sensi[:, :] = -np.inf

        outputs[0][0] = np.squeeze(sensi)


# TODO: Warning! This is not passing tests.
class runCkineKineticOp(Op):
    itypes = [dvector]
    otypes = [dvector]

    def __init__(self, ts, condense):
        assert condense.size == 56

        self.ts = ts
        self.condense = condense

    def infer_shape(self, node, i0_shapes):
        assert len(i0_shapes) == 1
        return [(self.ts.size, )]

    def perform(self, node, inputs, outputs):
        yOut, retVal = runCkineU(self.ts, inputs[0])

        assert yOut.shape == (self.ts.size, 56)

        if retVal < 0:
            yOut[:] = -np.inf

        outputs[0][0] = np.dot(yOut, self.condense)

    def grad(self, inputs, g):
        """ Calculate the runCkineOp gradient. """
        return [dot(g[0], runCkineOpKineticDiff(self.ts, self.condense)(inputs[0]))]


class runCkineOpKineticDiff(Op):
    itypes = [dvector]
    otypes = [dmatrix]

    def __init__(self, ts, condense):
        assert condense.size == 56

        self.ts = ts
        self.condense = condense

    def perform(self, node, inputs, outputs):
        _, retVal, sensi = runCkineSensi(self.ts, inputs[0])

        if retVal < 0:
            sensi[:, :] = -np.inf

        outputs[0][0] = np.dot(np.transpose(sensi), self.condense)
