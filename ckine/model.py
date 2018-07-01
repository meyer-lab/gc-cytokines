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


def runCkine (tps, rxn, tfr):
    """ Wrapper if rxn and tfr are separate. """
    return runCkineU (tps, np.concatenate((rxn, tfr)))


def runCkineU (tps, rxntfr):
    global libb

    assert rxntfr.size == 24
    assert rxntfr[15] < 1.0 # Check that sortF won't throw

    yOut = np.zeros((tps.size, 48), dtype=np.float64)

    retVal = libb.runCkine(tps.ctypes.data_as(ct.POINTER(ct.c_double)),
                           tps.size,
                           yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
                           rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)),
                           False,
                           ct.POINTER(ct.c_double)())

    return (yOut, retVal)


def runCkineSensi (tps, rxntfr):
    global libb

    assert rxntfr.size == 24

    yOut = np.zeros((tps.size, 48), dtype=np.float64)

    sensV = np.zeros((48, 24, tps.size), dtype=np.float64, order='F')

    retVal = libb.runCkine(tps.ctypes.data_as(ct.POINTER(ct.c_double)),
                           tps.size,
                           yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
                           rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)),
                           True,
                           sensV.ctypes.data_as(ct.POINTER(ct.c_double)))

    return (yOut, retVal, sensV)


def dy_dt(y, t, rxn):
    global libb

    assert rxn.size == 13

    rxntfr = np.concatenate((rxn, np.ones(15, dtype=np.float64)*0.9))

    yOut = np.zeros_like(y)

    libb.dydt_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t,
                yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)))

    return yOut


def jacobian(y, t, rxn):
    global libb

    assert rxn.size == 13

    yOut = np.zeros((22, 22)) # size of the Jacobian matrix

    libb.jacobian_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_double(t), yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxn.ctypes.data_as(ct.POINTER(ct.c_double)))

    return yOut


def fullJacobian(y, t, rxn): # will eventually have to add tfR as an argument once we add more to fullJacobian
    global libb

    assert rxn.size == 24

    yOut = np.zeros((48, 48)) # size of the full Jacobian matrix

    libb.fullJacobian_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_double(t), yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxn.ctypes.data_as(ct.POINTER(ct.c_double)))
    return yOut

def fullModel(y, t, rxn, tfr):
    global libb

    rxntfr = np.concatenate((rxn, tfr))

    assert rxntfr.size == 24

    yOut = np.zeros_like(y)

    libb.fullModel_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t,
                     yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)))

    return yOut


__active_species_IDX = np.zeros(22, dtype=np.bool)
__active_species_IDX[np.array([7, 8, 14, 15, 18, 21])] = 1


def solveAutocrine(trafRates):
    """Faster approach to solve for steady state by directly calculating the starting point without needing odeint."""
    y0 = np.zeros(48 , np.float64)

    recIDX = np.array([0, 1, 2, 9, 16, 19], np.int)

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
    y0[recIDX + 22] = expr / kDeg / internalFrac
    y0[recIDX] = (expr + kRec*y0[recIDX + 22]*internalFrac)/endo

    return y0


def solveAutocrineComplete(rxnRates, trafRates):
    """This function determines the starting point for odeint. It runs the model for a really long time with no cytokine present to come to some steady state."""
    rxnRates = rxnRates.copy()
    autocrineT = np.array([0.0, 100000.0])

    y0 = np.zeros(48, np.float64)

    # For now assume 0 autocrine ligand
    rxnRates[0:4] = 0.0

    full_lambda = lambda y, t: fullModel(y, t, rxnRates, trafRates)

    yOut = odeint(full_lambda, y0, autocrineT, mxstep=int(1E5))

    return yOut[1, :]


def getActiveSpecies():
    """ Return a vector that indicates which species are active. """
    return __active_species_IDX


def getCytokineSpecies():
    """ Returns a list of vectors for which species are bound to which cytokines. """
    return list((np.arange(3, 9), np.arange(10, 16), np.arange(17, 19), np.arange(20, 22)))

def getSurfaceIL2RbSpecies():
    """ Returns a list of vectors for which surface species contain the IL2Rb receptor. """
    condense = np.zeros(48)
    condense[np.array([1, 4, 5, 7, 8, 11, 12, 14, 15])] = 1
    return condense


def getActiveCytokine(cytokineIDX, yVec):
    """ Get amount of active species. """
    assert len(yVec) == 22
    return np.sum((yVec * getActiveSpecies())[getCytokineSpecies()[cytokineIDX]])


def getTotalActiveCytokine(cytokineIDX, yVec):
    """ Get amount of surface and endosomal active species. """
    return getActiveCytokine(cytokineIDX, yVec[0:22]) + getActiveCytokine(cytokineIDX, yVec[22:22*2])

def surfaceReceptors(y):
    """This function takes in a vector y and returns the amounts of the 6 surface receptors"""
    IL2Ra = np.sum(y[np.array([0, 3, 5, 6, 8])])
    IL2Rb = np.sum(y[np.array([1, 4, 5, 7, 8, 11, 12, 14, 15])])
    gc = np.sum(y[np.array([2, 6, 7, 8, 13, 14, 15, 18, 21])])
    IL15Ra = np.sum(y[np.array([9, 10, 12, 13, 15])])
    IL7Ra = np.sum(y[np.array([16, 17, 18])])
    IL9R = np.sum(y[np.array([19, 20, 21])])
    return np.array([IL2Ra, IL2Rb, gc, IL15Ra, IL7Ra, IL9R])

def totalReceptors(yVec):
    """This function takes in a vector y and returns the amounts of all 6 receptors in both cell compartments"""
    return surfaceReceptors(yVec) + surfaceReceptors(yVec[22:44])
