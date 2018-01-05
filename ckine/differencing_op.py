import numpy as np, theano.tensor as T
from concurrent.futures import ProcessPoolExecutor, Future, Executor
from threading import Lock


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

        if parallel:
            self.parallel = True
            self.pool = ProcessPoolExecutor()
        else:
            self.parallel = False
            self.pool = DummyExecutor()

        self.dg = centralDiffGrad(calcModel, self.pool)

    def infer_shape(self, node, i0_shapes):
        return [(self.M.concs*2, )]

    def perform(self, node, inputs, outputs):
        vec, = inputs
        mu = self.M.calc(vec, pool=self.pool)

        if np.any(np.isclose(mu, -100.)):
            raise RuntimeError("Activity calculation failed.")

        outputs[0][0] = np.array(mu)

    def grad(self, inputs, g):
        graddf = self.dg(inputs[0])

        return [T.dot(g[0], graddf)]


class centralDiffGrad(T.Op):
    itypes = [T.dvector]
    otypes = [T.dmatrix]

    def __init__(self, calcModel, pool=None):
        self.M = calcModel

        # Handle no pool being passed
        if pool is None:
            self.pool = DummyExecutor()
        else:
            self.pool = pool

    def perform(self, node, inputs, outputs):
        # Find our current point
        x0 = inputs[0]
        f0 = self.M.calc(x0, self.pool)

        if np.any(np.isclose(f0, -100.)):
            raise RuntimeError("Activity calculation failed so not able to evaluate gradient.")

        epsilon = 1.0E-7

        output = list()

        jac = np.empty((x0.size, f0.size), dtype=np.float64)

        # Schedule all the calculations
        for i in range(x0.size):
            dx = x0.copy()
            dx[i] = dx[i] + epsilon
            output.append(self.M.calc_schedule(dx, self.pool))

        # Process all the results
        for i, item in enumerate(output):
            jac[i] = (self.M.calc_reduce(item) - f0)/epsilon

        outputs[0][0] = np.transpose(jac)
