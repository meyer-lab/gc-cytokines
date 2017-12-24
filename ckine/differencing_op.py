import theano.tensor as T
import numpy as np
from concurrent.futures import ProcessPoolExecutor


class centralDiff(T.Op):
    itypes = [T.dvector]
    otypes = [T.dvector]

    def __init__(self, calcModel, parallel=True):
        self.M = calcModel

        if parallel:
            self.parallel = True
            self.pool = ProcessPoolExecutor()
        else:
            self.parallel = False
            self.pool = None

        self.dg = centralDiffGrad(calcModel, self.pool)

    def infer_shape(self, node, i0_shapes):
        return [(self.M.concs*2, )]

    def perform(self, node, inputs, outputs):
        vec, = inputs
        mu = self.M.calc(vec, pool=self.pool)

        outputs[0][0] = np.array(mu)

    def grad(self, inputs, g):
        graddf = self.dg(inputs[0])

        return [T.dot(g[0], graddf)]


class centralDiffGrad(T.Op):
    itypes = [T.dvector]
    otypes = [T.dmatrix]

    def __init__(self, calcModel, pool=None):
        self.M = calcModel
        self.pool = pool

    def perform(self, node, inputs, outputs):
        # Find our current point
        x0 = inputs[0]
        f0 = self.M.calc(x0)
        epsilon = 1.0E-7

        output = list()

        jac = np.zeros((len(x0), len(f0)), dtype=np.float64)

        if self.pool is not None:
            for i in range(len(x0)):
                dx = x0.copy()
                dx[i] = dx[i] + epsilon
                output.append(self.pool.submit(self.M.calc, dx))

            for i, item in enumerate(output):
                jac[i] = (item.result() - f0)/epsilon
        else:
            for i in range(len(x0)):
                dx = x0.copy()
                dx[i] = dx[i] + epsilon
                jac[i] = (self.M.calc(dx) - f0)/epsilon

        outputs[0][0] = np.transpose(jac)
