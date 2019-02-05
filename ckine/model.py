"""
A file that includes the model and important helper functions.
"""
import os
import ctypes as ct
import numpy as np


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine.so")
libb = ct.cdll.LoadLibrary(filename)
libb.fullModel_C.argtypes = (ct.POINTER(ct.c_double), ct.c_double, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
libb.runCkine.argtypes = (ct.POINTER(ct.c_double), ct.c_uint, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_bool, ct.c_double, ct.POINTER(ct.c_double))
libb.runCkineParallel.argtypes = (ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_uint, ct.c_uint, ct.POINTER(ct.c_double), ct.c_double, ct.POINTER(ct.c_double))
libb.runCkineS.argtypes = (ct.POINTER(ct.c_double), ct.c_uint, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_bool, ct.c_double, ct.POINTER(ct.c_double))
libb.runCkineSParallel.argtypes = (ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_uint, ct.c_uint, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_double, ct.POINTER(ct.c_double))

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

__internalV = 623.0 # endosomal volume
def internalV():
    """ Returns __internalV. """
    return __internalV

__nRxn = 17
def nRxn():
    """ Returns the length of the rxn rates vector (doesn't include traf rates). """
    return __nRxn


def runCkineU (tps, rxntfr, preT=0.0, prestim=None):
    """ Standard version of solver that returns species abundances given times and unknown rates. """
    rxntfr = rxntfr.copy()
    assert rxntfr.size == __nParams
    assert rxntfr[19] < 1.0 # Check that sortF won't throw

    yOut = np.zeros((tps.size, __nSpecies), dtype=np.float64)

    if preT != 0.0:
        assert preT > 0.0
        assert prestim.size == 6
        prestim = prestim.ctypes.data_as(ct.POINTER(ct.c_double))

    retVal = libb.runCkine(tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size,
                           yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
                           rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)), False, preT, prestim)

    return (yOut, retVal)


def runCkineS (tps, rxntfr, condense):
    """ Standard version of solver that returns species abundances given times and unknown rates. """
    rxntfr = rxntfr.copy()
    assert rxntfr.size == __nParams
    assert rxntfr[19] < 1.0 # Check that sortF won't throw

    assert condense.size == __nSpecies

    yOut = np.zeros((tps.size), dtype=np.float64)
    sensV = np.zeros((tps.size, __nParams), dtype=np.float64, order='C')

    retVal = libb.runCkineS(tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size,
                            yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
                            sensV.ctypes.data_as(ct.POINTER(ct.c_double)),
                            condense.ctypes.data_as(ct.POINTER(ct.c_double)),
                            rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)),
                            False, 0.0, None)

    return (yOut, retVal, sensV)


def runCkineU_IL2 (tps, rxntfr):
    """ Standard version of solver that returns species abundances given times and unknown rates. """
    rxntfr = rxntfr.copy()
    assert rxntfr.size == 10

    yOut = np.zeros((tps.size, __nSpecies), dtype=np.float64)

    retVal = libb.runCkine(tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size,
                           yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
                           rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)), True, 0.0, None)

    return (yOut, retVal)


def runIL2simple(input, IL, CD25=1.0):
    """ Version to focus on IL2Ra/Rb affinity adjustment. """
    # TODO: Update parameters based on distinct endosomal fitting.
    tps = np.array([500.0])

    # IL, kfwd, k1rev, k2rev, k4rev, k5rev, k11rev, R, R, R
    rxntfr = np.array([IL, 0.00449, 0.6*10*input[0],
                       0.6*144*input[1], 8.6677, 0.1233,
                       63.0 * 0.1233 / 1.5 * input[1], 3.8704*CD25, 0.734, 1.7147])

    yOut, retVal = runCkineU_IL2(tps, rxntfr)

    assert retVal == 0

    active = getTotalActiveCytokine(0, np.squeeze(yOut))

    return active


def runCkineUP (tps, rxntfr, preT=0.0, prestim=None):
    """ Version of runCkine that runs in parallel. """
    tps = np.array(tps)
    assert rxntfr.size % __nParams == 0
    assert rxntfr.shape[1] == __nParams

    assert (rxntfr[:, 19] < 1.0).all() # Check that sortF won't throw
    assert np.all(np.any(rxntfr > 0.0, axis=1)) # make sure at least one element is >0 for all rows

    yOut = np.zeros((rxntfr.shape[0]*tps.size, __nSpecies), dtype=np.float64)

    if preT != 0.0:
        assert preT > 0.0
        assert prestim.size == 6
        prestim = prestim.ctypes.data_as(ct.POINTER(ct.c_double))

    retVal = libb.runCkineParallel(rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)),
                                   tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size, rxntfr.shape[0],
                                   yOut.ctypes.data_as(ct.POINTER(ct.c_double)), preT, prestim)

    return (yOut, retVal)


def runCkineSP (tps, rxntfr, actV, preT=0.0, prestim=None):
    """ Version of runCkine that runs in parallel. """
    tps = np.array(tps)
    assert rxntfr.size % __nParams == 0
    assert rxntfr.shape[1] == __nParams
    assert (rxntfr[:, 19] < 1.0).all() # Check that sortF won't throw

    yOut = np.zeros((rxntfr.shape[0]*tps.size), dtype=np.float64)
    sensV = np.zeros((rxntfr.shape[0]*tps.size, __nParams), dtype=np.float64, order='C')

    if preT != 0.0:
        assert preT > 0.0
        assert prestim.size == 6
        prestim = prestim.ctypes.data_as(ct.POINTER(ct.c_double))

    retVal = libb.runCkineSParallel(rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)),
                                    tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size, rxntfr.shape[0],
                                    yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
                                    sensV.ctypes.data_as(ct.POINTER(ct.c_double)),
                                    actV.ctypes.data_as(ct.POINTER(ct.c_double)), preT, prestim)

    return (yOut, retVal, sensV)


def fullModel(y, t, rxntfr):
    """ Implement the full model based on dydt, trafficking, expression. """
    assert rxntfr.size == __nParams

    yOut = np.zeros_like(y)

    libb.fullModel_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t,
                     yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)))

    return yOut


__active_species_IDX = np.zeros(__halfL, dtype=np.bool)
__active_species_IDX[np.array([7, 8, 14, 15, 18, 21, 24, 27])] = 1


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
    return ((yVec * getActiveSpecies())[getCytokineSpecies()[cytokineIDX]]).sum()


def getTotalActiveCytokine(cytokineIDX, yVec):
    """ Get amount of surface and endosomal active species. """
    assert yVec.ndim == 1
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

def ligandDeg_IL2(yVec, sortF, kDeg):
    """ This function calculates rate of IL-2's total degradation. """
    # all indices are shifted by __halfL in order to get endosomal species
    yVec = yVec[__halfL::].copy()
    sum_inactive = np.sum(yVec[3:7]) * sortF # indexes 3-6 have IL2 bound but are inactive, only inactive species deal with sortF
    sum_active = np.sum(yVec[7:9]) # indices 7,8 have IL2 bound and are active
    return kDeg * (((sum_inactive + sum_active) * __internalStrength) + (yVec[__halfL] * __internalV)) # can assume all free ligand and active species are degraded at rate kDeg
