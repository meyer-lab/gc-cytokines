"""
Theano Op for using differencing for Jacobian calculation.
"""
from threading import Lock
import numpy as np, theano.tensor as T
from concurrent.futures import ProcessPoolExecutor, Future, Executor


class DummyExecutor(Executor):
    """
    Dummy executor to allow futures even when we're not running in parallel.
    """
    def __init__(self):
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True


class centralDiff(T.Op):
    itypes = [T.dvector]
    otypes = [T.dvector]

    def __init__(self, calcModel, parallel=True):
        self.M = calcModel
        self.dg = centralDiffGrad(calcModel, parallel)

    def perform(self, node, inputs, outputs):
        if np.any(np.greater(inputs[0], 1.0E4)):
            mu = np.full((self.M.concs*2, ), -np.inf, dtype=np.float64)
        else:
            mu = self.M.calc(inputs[0])

        outputs[0][0] = np.array(mu)

    def grad(self, inputs, g):
        """ Calculate the centralDiff gradient. """
        return [T.dot(g[0], self.dg(inputs[0]))]


class centralDiffGrad(T.Op):
    itypes = [T.dvector]
    otypes = [T.dmatrix]

    def __init__(self, calcModel, parallel):
        self.M = calcModel

        # Setup process pool if desired
        if parallel:
            self.pool = ProcessPoolExecutor()
        else:
            self.pool = DummyExecutor()

    def perform(self, node, inputs, outputs):
        # Find our current point
        x0 = inputs[0]

        epsilon = 1.0E-7

        output = list()

        jac = np.empty((x0.size, self.M.concs*2), dtype=np.float64)

        if np.any(np.greater(inputs[0], 1.0E4)):
            jac.fill(-np.inf)
        else:
            f0 = self.M.calc(x0)

            # Schedule all the calculations
            for i in range(x0.size):
                dx = x0.copy()
                dx[i] = dx[i] + epsilon
                output.append(self.pool.submit(self.M.calc, dx))

            # Process all the results
            for i, item in enumerate(output):
                jac[i] = (item.result() - f0)/epsilon

        outputs[0][0] = np.transpose(jac)
