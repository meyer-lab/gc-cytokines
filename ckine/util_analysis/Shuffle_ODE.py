import numpy as np


def approx_jacobian(func, y, delta=1.0E-9):
    """Approximate the Jacobian matrix of callable function func

       * Parameters
         None

       * Returns
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

       * Notes
         The approximation is done using forward differences. func in this case is the fullModel function from the main model file.

    """
    f0 = func(y)
    jac = np.zeros([y.size, f0.size])
    dy = np.zeros(y.size)

    for i in range(y.size):
        dy[i] = delta
        jac[i] = (func(y + dy) - f0)/delta
        dy[i] = 0.0

    return jac.transpose()
