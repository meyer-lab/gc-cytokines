"""
Theano Op for using differencing for Jacobian calculation.
"""
import concurrent.futures
import numpy as np
from theano.tensor import dot, dmatrix, dvector, Op
from .model import runCkineU, nSpecies, nParams, runCkineUP, runCkinePreT

pool = concurrent.futures.ThreadPoolExecutor()

class runCkineOp(Op):
    itypes, otypes = [dvector], [dvector]

    def __init__(self, ts):
        self.dOp = runCkineOpDiff(ts)

    def infer_shape(self, node, i0_shapes):
        assert len(i0_shapes) == 1
        return [(nSpecies(), )]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.dOp.runCkine(inputs, False)

    def grad(self, inputs, g):
        """ Calculate the runCkineOp gradient. """
        return [dot(g[0], self.dOp(inputs[0]))]


class runCkineOpDiff(Op):
    itypes, otypes = [dvector], [dmatrix]

    def __init__(self, ts):
        if ts.size > 1:
            raise NotImplementedError('This Op only works with a single time point.')

        self.ts = ts

    def runCkine(self, inputs, sensi):
        outt = runCkineU(self.ts, inputs[0], sensi)
        assert outt[1] >= 0
        assert outt[0].size == nSpecies()

        if sensi is True:
            return np.squeeze(outt[2])

        return np.squeeze(outt[0])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.runCkine(inputs, True)


class runCkinePreSOp(Op):
    itypes, otypes = [dvector], [dvector]

    def __init__(self, tpre, ts, postlig):
        self.dOp = runCkinePreSOpDiff(tpre, ts, postlig)

    def infer_shape(self, node, i0_shapes):
        assert len(i0_shapes) == 1
        return [(nSpecies(), )]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.dOp.runCkine(inputs[0])

    def grad(self, inputs, g):
        """ Calculate the runCkineOp gradient. """
        return [dot(g[0], self.dOp(inputs[0]))]


class runCkinePreSOpDiff(Op):
    itypes, otypes = [dvector], [dmatrix]

    def __init__(self, tpre, ts, postlig):
        assert ts.size == 1
        assert postlig.size == 6

        self.ts = ts
        self.tpre = tpre
        self.postlig = postlig

    def runCkine(self, inputs):
        outt = runCkinePreT(self.tpre, self.ts, inputs, self.postlig)

        assert outt[1] >= 0
        assert outt[0].size == nSpecies()

        return np.squeeze(outt[0])

    def perform(self, node, inputs, outputs):
        x0 = inputs[0]
        f0 = self.runCkine(x0)
        epsilon = 1.0E-7

        jac = np.zeros((len(x0), len(f0)), dtype=np.float64)
        dxl = list()

        for i in range(len(x0)):
            dx = x0.copy()
            dx[i] = dx[i] + epsilon
            dxl.append(dx)

        dxlr = pool.map(self.runCkine, dxl)

        for i, item in enumerate(dxlr):
            jac[i] = (item - f0)/epsilon

        outputs[0][0] = np.transpose(jac)


class runCkineKineticOp(Op):
    itypes, otypes = [dvector], [dvector]

    def __init__(self, ts, condense):
        self.dOp = runCkineOpKineticDiff(ts, condense)

    def infer_shape(self, node, i0_shapes):
        assert len(i0_shapes) == 1
        return [(self.dOp.ts.size, )]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.dOp.runCkine(inputs, False)

    def grad(self, inputs, g):
        """ Calculate the runCkineOp gradient. """
        return [dot(g[0], self.dOp(inputs[0]))]


class runCkineOpKineticDiff(Op):
    itypes, otypes = [dvector], [dmatrix]

    def __init__(self, ts, condense):
        self.ts = ts
        self.condense = condense
        assert condense.size == nSpecies()

    def runCkine(self, inputs, sensi):
        outt = runCkineU(self.ts, inputs[0], sensi)
        assert outt[0].shape == (self.ts.size, nSpecies())
        assert outt[1] >= 0

        if sensi is True:
            return np.dot(np.transpose(outt[2]), self.condense)

        return np.dot(outt[0], self.condense)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.runCkine(inputs, sensi=True)


class runCkineDoseOp(Op):
    itypes, otypes = [dvector], [dvector]

    def __init__(self, tt, condense, conditions):
        self.dOp = runCkineOpDoseDiff(tt, condense, conditions)

    def infer_shape(self, node, i0_shapes):
        assert len(i0_shapes) == 1
        return [(self.dOp.conditions.shape[0], )]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.dOp.runCkine(inputs, sensi=False)

    def grad(self, inputs, g):
        """ Calculate the runCkineOp gradient. """
        return [dot(g[0], self.dOp(inputs[0]))]


class runCkineOpDoseDiff(Op):
    itypes, otypes = [dvector], [dmatrix]

    def __init__(self, tt, condense, conditions):
        self.ts, self.condense, self.conditions = tt, condense, conditions
        assert tt.size == 1
        assert condense.size == nSpecies() # Check that we're condensing a species vector
        assert conditions.shape[1] == 6 # Check that this is a matrix of ligands

    def runCkine(self, inputs, sensi):
        assert inputs[0].size == nParams() - self.conditions.shape[1]
        rxntfr = np.reshape(np.tile(inputs[0], self.conditions.shape[0]), (self.conditions.shape[0], -1))
        rxntfr = np.concatenate((self.conditions, rxntfr), axis=1)

        outt = runCkineUP(self.ts, rxntfr, sensi)
        assert outt[0].shape == (self.conditions.shape[0], nSpecies())
        assert outt[1] >= 0

        if sensi is True:
            # We override the ligands, so don't pass along their gradient
            return np.dot(np.transpose(outt[2][:, 6::]), self.condense)

        return np.dot(outt[0], self.condense)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.runCkine(inputs, sensi=True)
