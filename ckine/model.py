import os, numpy as np, ctypes as ct
from scipy.integrate import odeint


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine.so")
libb = ct.cdll.LoadLibrary(filename)
libb.wrapper.argtypes = (ct.POINTER(ct.c_double), ct.c_double,
                         ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                         ct.POINTER(ct.c_double), ct.POINTER(ct.c_bool))


def wrapper(y, t, rxn, tfr, wrapIDX):
    global libb

    assert(len(y) == np.sum(wrapIDX))

    yOut = np.zeros_like(y)

    libb.wrapper(y.ctypes.data_as(ct.POINTER(ct.c_double)), t,
                 yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxn.ctypes.data_as(ct.POINTER(ct.c_double)),
                 tfr.ctypes.data_as(ct.POINTER(ct.c_double)), wrapIDX.ctypes.data_as(ct.POINTER(ct.c_double)))
    
    return yOut


__active_species_IDX = np.zeros(26, dtype=np.bool)
__active_species_IDX[np.array([8, 9, 16, 17, 21, 25])] = 1


__IL2_assoc = np.zeros(56, dtype=np.bool)
__IL2_assoc[0:10] = 1
__IL2_assoc[26:36] = 1
__IL2_assoc[52] = 1


def printModel(rxnRates, trafRates):
    # endo, activeEndo, sortF, kRec, kDeg
    print("Endocytosis: " + str(trafRates[0]))
    print("activeEndo: " + str(trafRates[1]))
    print("sortF: " + str(trafRates[2]))
    print("kRec: " + str(trafRates[3]))
    print("kDeg: " + str(trafRates[4]))
    print("Receptor expression: " + str(trafRates[5:11]))
    print(".....Reaction rates.....")
    print("IL2: " + str(rxnRates[0]))
    print("IL15: " + str(rxnRates[1]))
    print("IL7: " + str(rxnRates[2]))
    print("IL9: " + str(rxnRates[3]))
    print("kfwd: " + str(rxnRates[4]))
    print("k5rev: " + str(rxnRates[5]))
    print("k6rev: " + str(rxnRates[6]))
    print(rxnRates[7::])


def solveAutocrine(trafRates):
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


"""
def solveAutocrineComplete(rxnRates, trafRates):
    rxnRates = rxnRates.copy()
    autocrineT = np.array([0.0, 100000.0])

    y0 = np.zeros(26*2 + 4, np.float64)

    # For now assume 0 autocrine ligand
    # TODO: Consider handling autocrine ligand more gracefully
    rxnRates[0:4] = 0.0

    full_lambda = lambda y, t: fullModel(y, t, rxnRates, trafRates, __active_species_IDX)

    yOut = odeint(full_lambda, y0, autocrineT, mxstep=int(1E5))

    return yOut[1, :]
"""


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
