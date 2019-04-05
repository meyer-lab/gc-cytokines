"""
Theano Op for using differencing for Jacobian calculation.
"""
import numpy as np
from theano.tensor import dot, dmatrix, dvector, Op
from .model import nSpecies, nParams, runCkineUP, runCkineSP


class runCkineDoseOp(Op):
    """ Runs model for a dose response at a single time point. """
    itypes, otypes = [dvector], [dvector]

    def __init__(self, tt, condense, conditions, preT=0.0, prestim=None):
        self.dOp = runCkineOpDoseDiff(tt, condense, conditions, preT, prestim)

    def infer_shape(self, node, i0_shapes):
        """ infering shape """
        assert len(i0_shapes) == 1
        return [(self.dOp.conditions.shape[0] * self.dOp.ts.size, )]

    def perform(self, node, inputs, outputs, params=None):
        outputs[0][0] = self.dOp.runCkine(inputs, sensi=False)

    def grad(self, inputs, g):
        """ Calculate the runCkineOp gradient. """
        return [dot(g[0], self.dOp(inputs[0]))]


class runCkineOpDoseDiff(Op):
    """ Gradient of model for a dose response at a single time point. """
    itypes, otypes = [dvector], [dmatrix]

    def __init__(self, tt, condense, conditions, preT, prestim):
        self.ts, self.condense, self.conditions, self.preT, self.prestim = tt, condense, conditions, preT, prestim

        if preT > 0.0:
            assert prestim.size == 6

        assert condense.size == nSpecies()  # Check that we're condensing a species vector
        assert conditions.shape[1] == 6  # Check that this is a matrix of ligands

    def runCkine(self, inputs, sensi):
        """ function for runCkine """
        assert inputs[0].size == nParams() - self.conditions.shape[1]
        rxntfr = np.reshape(np.tile(inputs[0], self.conditions.shape[0]), (self.conditions.shape[0], -1))
        rxntfr = np.concatenate((self.conditions, rxntfr), axis=1)

        if sensi is False:
            outt = runCkineUP(self.ts, rxntfr, self.preT, self.prestim)
            assert outt[1] >= 0
            return np.dot(outt[0], self.condense)

        outt = runCkineSP(self.ts, rxntfr, self.condense, self.preT, self.prestim)
        assert outt[0].shape == (self.conditions.shape[0] * self.ts.size, )
        assert outt[1] >= 0

        # We override the ligands, so don't pass along their gradient
        return outt[2][:, 6::]

    def perform(self, node, inputs, outputs, params=None):
        outputs[0][0] = self.runCkine(inputs, sensi=True)
