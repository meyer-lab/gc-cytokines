from ckine.model import fullModel, dy_dt
import numpy as np, scipy


def approx_jacobian():
    """Approximate the Jacobian matrix of callable function func

       * Parameters
         None

       * Returns
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

       * Notes
         The approximation is done using forward differences. func in this case is the fullModel function from the main model file. 

    """
    trafRates = np.random.sample(11)
    rxnRates = np.random.sample(15)
    x0 = np.random.sample(56)
    f0 = fullModel(x0, 0.0 , rxnRates, trafRates)
    jac = np.zeros([len(x0),len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = np.sqrt(np.finfo(float).eps)
        jac[i] = (fullModel(x0 + dx, 0.0 , rxnRates, trafRates) - f0)/(np.sqrt(np.finfo(float).eps))
        dx[i] = 0.0
        
    return jac.transpose()


def approx_jac_dydt(y, t, rxn, delta=1.0E-9):
    """Approximate the Jacobian matrix of callable function func

       * Returns
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

       * Notes
         The approximation is done using forward differences. func in this case is the fullModel function from the main model file. 

    """
    f0 = dy_dt(y, t, rxn)
    jac = np.zeros([y.size, f0.size])

    for i in range(y.size):
        dy = y.copy()
        dy[i] = dy[i] + delta

        jac[i] = (dy_dt(dy, t, rxn) - f0)/delta
        
    return jac.transpose()
