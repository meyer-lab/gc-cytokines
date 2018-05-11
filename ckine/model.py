"""
A file that includes the model and important helper functions.
"""
import os
import numpy as np
import ctypes as ct
from scipy.integrate import odeint


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine.so")
libb = ct.cdll.LoadLibrary(filename)
libb.dydt_C.argtypes = (ct.POINTER(ct.c_double), ct.c_double,
                        ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
libb.jacobian_C.argtypes = (ct.POINTER(ct.c_double), ct.c_double,
                        ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
libb.fullModel_C.argtypes = (ct.POINTER(ct.c_double), ct.c_double,
                             ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
libb.runCkine.argtypes = (ct.POINTER(ct.c_double), ct.c_uint,
                          ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                          ct.c_bool, ct.POINTER(ct.c_double))


def runCkine (tps, rxn, tfr):
    """ Wrapper if rxn and tfr are separate. """
    return runCkineU (tps, np.concatenate((rxn, tfr)))


def runCkineU (tps, rxntfr):
    global libb

    assert(rxntfr.size == 25)

    yOut = np.zeros((tps.size, 56), dtype=np.float64)

    retVal = libb.runCkine(tps.ctypes.data_as(ct.POINTER(ct.c_double)),
                           tps.size,
                           yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
                           rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)),
                           False,
                           ct.POINTER(ct.c_double)())

    return (yOut, retVal)


def runCkineSensi (tps, rxntfr):
    global libb

    assert(rxntfr.size == 25)

    yOut = np.zeros((tps.size, 56), dtype=np.float64)

    sensV = np.zeros((56, 25, tps.size), dtype=np.float64, order='F')

    retVal = libb.runCkine(tps.ctypes.data_as(ct.POINTER(ct.c_double)),
                           tps.size,
                           yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
                           rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)),
                           True,
                           sensV.ctypes.data_as(ct.POINTER(ct.c_double)))

    return (yOut, retVal, sensV)


def dy_dt(y, t, rxn):
    global libb

    assert(rxn.size == 14)

    yOut = np.zeros_like(y)

    libb.dydt_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t,
                yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxn.ctypes.data_as(ct.POINTER(ct.c_double)))
    
    return yOut

def jacobian(y, t, rxn):
    global libb
    
    assert(rxn.size == 14)
    
    yOut = np.zeros((26, 26)) # size of the Jacobian matrix
    
    libb.jacobian_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t,
                yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxn.ctypes.data_as(ct.POINTER(ct.c_double)))
    
    return yOut


def fullModel(y, t, rxn, tfr):
    global libb

    rxntfr = np.concatenate((rxn, tfr))

    assert(rxntfr.size == 25)

    yOut = np.zeros_like(y)

    libb.fullModel_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t,
                     yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)))
    
    return yOut


__active_species_IDX = np.zeros(26, dtype=np.bool)
__active_species_IDX[np.array([8, 9, 16, 17, 21, 25])] = 1


def solveAutocrine(trafRates):
    """Faster approach to solve for steady state by directly calculating the starting point without needing odeint."""
    y0 = np.zeros(26*2 + 4, np.float64)

    recIDX = np.array([0, 1, 2, 10, 18, 22], np.int)

    # Expr
    expr = trafRates[5:11]

    internalFrac = 0.5 # Same as that used in TAM model

    # Expand out trafficking terms
    endo, sortF, kRec, kDeg = trafRates[np.array([0, 2, 3, 4])]

    # Correct for sorting fraction
    kRec = kRec*(1-sortF)
    kDeg = kDeg*sortF

    # Assuming no autocrine ligand, so can solve steady state
    # Add the species
    y0[recIDX + 26] = expr / kDeg / internalFrac
    y0[recIDX] = (expr + kRec*y0[recIDX + 26]*internalFrac)/endo

    return y0


def solveAutocrineComplete(rxnRates, trafRates):
    """This function determines the starting point for odeint. It runs the model for a really long time with no cytokine present to come to some steady state."""
    rxnRates = rxnRates.copy()
    autocrineT = np.array([0.0, 100000.0])

    y0 = np.zeros(26*2 + 4, np.float64)

    # For now assume 0 autocrine ligand
    # TODO: Consider handling autocrine ligand more gracefully
    rxnRates[0:4] = 0.0

    full_lambda = lambda y, t: fullModel(y, t, rxnRates, trafRates)

    yOut = odeint(full_lambda, y0, autocrineT, mxstep=int(1E5))

    return yOut[1, :]


def getActiveSpecies():
    """ Return a vector that indicates which species are active. """
    return __active_species_IDX


def getCytokineSpecies():
    """ Returns a list of vectors for which species are bound to which cytokines. """
    return list((np.arange(3, 10), np.arange(11, 18), np.arange(19, 22), np.arange(23, 26)))


def getActiveCytokine(cytokineIDX, yVec):
    """ Get amount of active species. """
    assert(len(yVec) == 26)
    return np.sum((yVec * getActiveSpecies())[getCytokineSpecies()[cytokineIDX]])


def getTotalActiveCytokine(cytokineIDX, yVec):
    """ Get amount of surface and endosomal active species. """
    return getActiveCytokine(cytokineIDX, yVec[0:26]) + getActiveCytokine(cytokineIDX, yVec[26:26*2])

def surfaceReceptors(y):
    """This function takes in a vector y and returns the amounts of the 6 surface receptors"""
    IL2Ra = np.sum(y[np.array([0, 3, 6, 7, 9])])
    IL2Rb = np.sum(y[np.array([1, 4, 6, 8, 9, 12, 14, 16, 17])])
    gc = np.sum(y[np.array([2, 5, 7, 8, 9, 13, 15, 16, 17, 20, 21, 24, 25])])
    IL15Ra = np.sum(y[np.array([10, 11, 14, 15, 17])])
    IL7Ra = np.sum(y[np.array([18, 19, 21])])
    IL9R = np.sum(y[np.array([22, 23, 25])])
    return np.array([IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R])

def totalReceptors(yVec):
    """This function takes in a vector y and returns the amounts of all 6 receptors in both cell compartments"""
    return surfaceReceptors(yVec) + surfaceReceptors(yVec[26:52])
