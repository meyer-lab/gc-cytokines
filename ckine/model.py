"""
A file that includes the model and important helper functions.
"""
import os
import ctypes as ct
import numpy as np
from scipy.integrate import odeint


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine.so")
libb = ct.cdll.LoadLibrary(filename)
libb.dydt_C.argtypes = (ct.POINTER(ct.c_double), ct.c_double, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
libb.jacobian_C.argtypes = (ct.POINTER(ct.c_double), ct.c_double, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
libb.fullModel_C.argtypes = (ct.POINTER(ct.c_double), ct.c_double, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
libb.runCkine.argtypes = (ct.POINTER(ct.c_double), ct.c_uint, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_bool, ct.POINTER(ct.c_double))
libb.runCkineParallel.argtypes = (ct.POINTER(ct.c_double), ct.c_double, ct.c_uint, ct.c_bool, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))


__nSpecies = 62
def nSpecies():
    """ Returns the total number of species in the model. """
    return __nSpecies

__halfL = 28
def halfL():
    """ Returns the number of species on the surface alone. """
    return __halfL

__nParams = 30
def nParams():
    """ Returns the length of the rxntfR vector. """
    return __nParams

__internalStrength = 0.5 # strength of endosomal activity relative to surface
def internalStrength():
    """Returns the internalStrength of endosomal activity."""
    return __internalStrength

__nRxn = 17
def nRxn():
    """ Returns the length of the rxn rates vector (doesn't include traf rates). """
    return __nRxn


def runCkineU (tps, rxntfr, sensi=False):
    assert rxntfr.size == __nParams
    assert rxntfr[19] < 1.0 # Check that sortF won't throw

    yOut = np.zeros((tps.size, __nSpecies), dtype=np.float64)

    if sensi is True:
        sensV = np.zeros((__nSpecies, __nParams, tps.size), dtype=np.float64, order='F')
        sensP = sensV.ctypes.data_as(ct.POINTER(ct.c_double))
    else:
        sensP = ct.POINTER(ct.c_double)()


    retVal = libb.runCkine(tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size,
                           yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
                           rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)),
                           sensi, sensP)

    if sensi is True:
        return (yOut, retVal, sensV)
    else:
        return (yOut, retVal)


def runCkineUP (tp, rxntfr, sensi=False):
    assert rxntfr.size % __nParams == 0
    assert rxntfr.shape[1] == __nParams
    assert (rxntfr[:, 19] < 1.0).all() # Check that sortF won't throw

    yOut = np.zeros((rxntfr.shape[0], __nSpecies), dtype=np.float64)

    if sensi is True:
        sensV = np.zeros((__nSpecies, __nParams, rxntfr.shape[0]), dtype=np.float64, order='F')
        sensP = sensV.ctypes.data_as(ct.POINTER(ct.c_double))
    else:
        sensP = ct.POINTER(ct.c_double)()

    retVal = libb.runCkineParallel(rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)), tp, rxntfr.shape[0], sensi,
                                   yOut.ctypes.data_as(ct.POINTER(ct.c_double)), sensP)

    if sensi is True:
        return (yOut, retVal, sensV)
    else:
        return (yOut, retVal)


def dy_dt(y, t, rxn):

    assert rxn.size == __nRxn
    rxntfr = np.concatenate((rxn, np.ones(13, dtype=np.float64)*0.9))

    yOut = np.zeros_like(y)

    libb.dydt_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t,
                yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)))

    return yOut


def jacobian(y, t, rxn):

    assert rxn.size == __nRxn

    # Pad with zeros so we don't get a sortF panic
    rxn = rxn.copy()
    rxn = np.concatenate((rxn, np.zeros(20, dtype=np.float64)))

    yOut = np.zeros((__halfL, __halfL)) # size of the Jacobian matrix for surface alone

    libb.jacobian_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_double(t), yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxn.ctypes.data_as(ct.POINTER(ct.c_double)))

    return yOut

def fullJacobian(y, t, rxntfR): # will eventually have to add tfR as an argument once we add more to fullJacobian
    assert rxntfR.size == __nParams

    yOut = np.zeros((__nSpecies, __nSpecies)) # size of the full Jacobian matrix

    libb.fullJacobian_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_double(t), yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfR.ctypes.data_as(ct.POINTER(ct.c_double)))
    return yOut

def fullModel(y, t, rxn, tfr):

    rxntfr = np.concatenate((rxn, tfr))
    assert rxntfr.size == __nParams

    yOut = np.zeros_like(y)

    libb.fullModel_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t,
                     yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)))

    return yOut


__active_species_IDX = np.zeros(__halfL, dtype=np.float64)
__active_species_IDX[np.array([7, 8, 14, 15, 18, 21, 24, 27])] = 1

def solveAutocrine(trafRates):
    """Faster approach to solve for steady state by directly calculating the starting point without needing odeint."""
    y0 = np.zeros(__nSpecies , np.float64)

    recIDX = np.array([0, 1, 2, 9, 16, 19, 22, 25], np.int)

    # Expr
    expr = trafRates[5:13]

    internalFrac = 0.5 # Same as that used in TAM model

    # Expand out trafficking terms
    endo, sortF, kRec, kDeg = trafRates[np.array([0, 2, 3, 4])]

    # Correct for sorting fraction
    kRec = kRec*(1-sortF)
    kDeg = kDeg*sortF

    # Assuming no autocrine ligand, so can solve steady state
    # Add the species
    y0[recIDX + __halfL] = expr / kDeg / internalFrac
    y0[recIDX] = (expr + kRec*y0[recIDX + __halfL]*internalFrac)/endo

    return y0


def solveAutocrineComplete(rxnRates, trafRates):
    """This function determines the starting point for odeint. It runs the model for a really long time with no cytokine present to come to some steady state."""
    rxnRates = rxnRates.copy()
    autocrineT = np.array([0.0, 100000.0])

    y0 = np.zeros(__nSpecies, np.float64)

    # For now assume 0 autocrine ligand
    rxnRates[0:6] = 0.0

    full_lambda = lambda y, t: fullModel(y, t, rxnRates, trafRates)

    yOut = odeint(full_lambda, y0, autocrineT, mxstep=int(1E5))

    return yOut[1, :]


def getActiveSpecies():
    """ Return a vector that indicates which species are active. """
    return __active_species_IDX


def getTotalActiveSpecies():
    """ Return a vector of all the species (surface + endosome) which are active. """
    activity = getActiveSpecies()
    return np.concatenate((activity, __internalStrength * activity, np.zeros(6)))

def getCytokineSpecies():
    """ Returns a list of vectors for which species are bound to which cytokines. """
    return list((np.arange(3, 9), np.arange(10, 16), np.arange(17, 19), np.arange(20, 22), np.arange(23, 25), np.arange(26, 28)))

def getSurfaceIL2RbSpecies():
    """ Returns a list of vectors for which surface species contain the IL2Rb receptor. """
    condense = np.zeros(__nSpecies)
    condense[np.array([1, 4, 5, 7, 8, 11, 12, 14, 15])] = 1
    return condense

def getSurfaceGCSpecies():
    """ Returns a list of vectors for which surface species contain the gc receptor. """
    condense = np.zeros(__nSpecies)
    condense[np.array([2, 6, 7, 8, 13, 14, 15, 18, 21])] = 1
    return condense


def getActiveCytokine(cytokineIDX, yVec):
    """ Get amount of active species. """
    assert len(yVec) == __halfL
    return np.sum((yVec * getActiveSpecies())[getCytokineSpecies()[cytokineIDX]])


def getTotalActiveCytokine(cytokineIDX, yVec):
    """ Get amount of surface and endosomal active species. """
    return getActiveCytokine(cytokineIDX, yVec[0:__halfL]) + __internalStrength * getActiveCytokine(cytokineIDX, yVec[__halfL:__halfL*2])


def surfaceReceptors(y):
    """This function takes in a vector y and returns the amounts of the 8 surface receptors"""
    IL2Ra = np.sum(y[np.array([0, 3, 5, 6, 8])])
    IL2Rb = np.sum(y[np.array([1, 4, 5, 7, 8, 11, 12, 14, 15])])
    gc = np.sum(y[np.array([2, 6, 7, 8, 13, 14, 15, 18, 21])])
    IL15Ra = np.sum(y[np.array([9, 10, 12, 13, 15])])
    IL7Ra = np.sum(y[np.array([16, 17, 18])])
    IL9R = np.sum(y[np.array([19, 20, 21])])
    IL4Ra = np.sum(y[np.array([22, 23, 24])])
    IL21Ra = np.sum(y[np.array([25, 26, 27])])
    return np.array([IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R, IL4Ra, IL21Ra])

def totalReceptors(yVec):
    """This function takes in a vector y and returns the amounts of all 8 receptors in both cell compartments"""
    return surfaceReceptors(yVec) + __internalStrength * surfaceReceptors(yVec[__halfL:__halfL*2])
