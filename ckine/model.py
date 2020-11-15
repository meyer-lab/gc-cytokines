"""
A file that includes the model and important helper functions.
"""
import os
from collections import OrderedDict
import ctypes as ct
import numpy as np


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine.so")
libb = ct.cdll.LoadLibrary(filename)
pcd = ct.POINTER(ct.c_double)
libb.fullModel_C.argtypes = (pcd, ct.c_double, pcd, pcd)
libb.runCkine.argtypes = (pcd, ct.c_uint, pcd, pcd, ct.c_bool, ct.c_double, pcd)
libb.runCkineParallel.argtypes = (pcd, pcd, ct.c_uint, ct.c_uint, pcd, ct.c_double, pcd)
libb.runCkineSParallel.argtypes = (pcd, pcd, ct.c_uint, ct.c_uint, pcd, pcd, pcd, ct.c_double, pcd)


def nSpecies():
    """ Returns the total number of species in the model. """
    return 62


def halfL():
    """ Returns the number of species on the surface alone. """
    return 28


def nParams():
    """ Returns the length of the rxntfR vector. """
    return 30


def rxParams():
    """ Returns the length of the rxntfR vector. """
    return 60


def internalStrength():
    """Returns the internalStrength of endosomal activity."""
    return 0.5


def runCkineUP(tps, rxntfr, preT=0.0, prestim=None, actV=None, mut_name=None):
    """ Version of runCkine that runs in parallel. If actV is set we'll return sensitivities. """
    rxntfr = np.atleast_2d(rxntfr)
    tps = np.array(tps)
    assert np.all(np.any(rxntfr > 0.0, axis=1)), "Make sure at least one element is >0 for all rows."
    assert not np.any(rxntfr < 0.0), "Make sure no values are negative."

    # Convert if we're using the condensed model
    rxnSizeStart = rxntfr.shape[1]
    if rxntfr.shape[1] == nParams():
        assert rxntfr.size % nParams() == 0
        assert (rxntfr[:, 19] < 1.0).all()  # Check that sortF won't throw

        rxntfr = getRateVec(rxntfr)
        if mut_name:
            rxntfr = mut_adjust(rxntfr, mutaff, mut_name)
    else:
        assert rxntfr.size % rxParams() == 0
        assert rxntfr.shape[1] == rxParams()

    if preT != 0.0:
        assert preT > 0.0
        assert prestim.size == 6
        prestim = prestim.ctypes.data_as(ct.POINTER(ct.c_double))

    if actV is None:
        yOut = np.zeros((rxntfr.shape[0] * tps.size, nSpecies()), dtype=np.float64)

        retVal = libb.runCkineParallel(
            rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)), tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size, rxntfr.shape[0], yOut.ctypes.data_as(ct.POINTER(ct.c_double)), preT, prestim
        )
    else:
        yOut = np.zeros((rxntfr.shape[0] * tps.size), dtype=np.float64)
        sensV = np.zeros((rxntfr.shape[0] * tps.size, rxParams()), dtype=np.float64, order="C")

        retVal = libb.runCkineSParallel(
            rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)),
            tps.ctypes.data_as(ct.POINTER(ct.c_double)),
            tps.size,
            rxntfr.shape[0],
            yOut.ctypes.data_as(ct.POINTER(ct.c_double)),
            sensV.ctypes.data_as(ct.POINTER(ct.c_double)),
            actV.ctypes.data_as(ct.POINTER(ct.c_double)),
            preT,
            prestim,
        )

    assert retVal >= 0  # make sure solver worked

    if actV is not None:
        # If we're using the condensed model
        if rxnSizeStart == nParams():
            sensV = condenseSENV(sensV)

        return (yOut, sensV)

    return yOut


def fullModel(y, t, rxntfr):
    """ Implement the full model based on dydt, trafficking, expression. """
    assert rxntfr.size == nParams()

    yOut = np.zeros_like(y)

    rxntfr = getRateVec(rxntfr)

    libb.fullModel_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t, yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)))

    return yOut


__active_species_IDX = np.zeros(halfL(), dtype=np.bool)
__active_species_IDX[np.array([7, 8, 14, 15, 18, 21, 24, 27])] = 1


def getActiveSpecies():
    """ Return a vector that indicates which species are active. """
    return __active_species_IDX


def getTotalActiveSpecies():
    """ Return a vector of all the species (surface + endosome) which are active. """
    activity = getActiveSpecies()
    return np.concatenate((activity, internalStrength() * activity, np.zeros(6)))


def getCytokineSpecies():
    """ Returns a list of vectors for which species are bound to which cytokines. """
    return list((np.arange(3, 9), np.arange(10, 16), np.arange(17, 19), np.arange(20, 22), np.arange(23, 25), np.arange(26, 28)))


def getSurfaceIL2RbSpecies():
    """ Returns a list of vectors for which surface species contain the IL2Rb receptor. """
    condense = np.zeros(nSpecies())
    condense[np.array([1, 4, 5, 7, 8, 11, 12, 14, 15])] = 1
    return condense


def getSurfaceGCSpecies():
    """ Returns a list of vectors for which surface species contain the gc receptor. """
    condense = np.zeros(nSpecies())
    condense[np.array([2, 6, 7, 8, 13, 14, 15, 18, 21])] = 1
    return condense


def getActiveCytokine(cytokineIDX, yVec):
    """ Get amount of active species. """
    return ((yVec * getActiveSpecies())[getCytokineSpecies()[cytokineIDX]]).sum()


def getTotalActiveCytokine(cytokineIDX, yVec):
    """ Get amount of surface and endosomal active species. """
    assert yVec.ndim == 1
    return getActiveCytokine(cytokineIDX, yVec[0:halfL()]) + internalStrength() * getActiveCytokine(cytokineIDX, yVec[halfL(): halfL() * 2])


def receptor_expression(receptor_abundance, endo, kRec, sortF, kDeg):
    """ Uses receptor abundance (from flow) and trafficking rates to calculate receptor expression rate at steady state. """
    rec_ex = (receptor_abundance * endo) / (1.0 + ((kRec * (1.0 - sortF)) / (kDeg * sortF)))
    return rec_ex


def condenseSENV(sensVin):
    """ Condense sensitivities down into the old rxnRates format. """
    sensVin[:, 7:27] += sensVin[:, 27:47] * 2.0
    sensVin[:, 10] += 12.0 * sensVin[:, 11] / 1.5 + 63.0 * sensVin[:, 12] / 1.5
    sensV = sensVin[:, np.array([0, 1, 2, 3, 4, 5, 6, 9, 10, 15, 16, 17, 18, 20, 22, 24, 26, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])]

    return sensV


def getparamsdict(rxntfr):
    """Where rate vectors and constants are defined, organized in an ordered dictionary"""
    rd = OrderedDict()

    kfbnd = 0.60
    rd["IL2"], rd["IL15"], rd["IL7"], rd["IL9"], rd["IL4"], rd["IL21"], rd["kfwd"] = tuple(rxntfr[0:7])
    rd["surf.k1rev"] = kfbnd * 10.0  # 7
    rd["surf.k2rev"] = kfbnd * 144.0
    rd["surf.k4rev"] = rxntfr[7]  # 9
    rd["surf.k5rev"] = rxntfr[8]  # 10
    rd["surf.k10rev"] = 12.0 * rd["surf.k5rev"] / 1.5
    rd["surf.k11rev"] = 63.0 * rd["surf.k5rev"] / 1.5
    rd["surf.k13rev"] = kfbnd * 0.065
    rd["surf.k14rev"] = kfbnd * 438.0
    rd["surf.k16rev"] = rxntfr[9]
    rd["surf.k17rev"] = rxntfr[10]  # 16
    rd["surf.k22rev"] = rxntfr[11]
    rd["surf.k23rev"] = rxntfr[12]
    rd["surf.k25rev"] = kfbnd * 59.0
    rd["surf.k27rev"] = rxntfr[13]
    rd["surf.k29rev"] = kfbnd * 0.1
    rd["surf.k31rev"] = rxntfr[14]
    rd["surf.k32rev"] = kfbnd * 1.0
    rd["surf.k33rev"] = rxntfr[15]
    rd["surf.k34rev"] = kfbnd * 0.07
    rd["surf.k35rev"] = rxntfr[16]

    for key, value in rd.copy().items():
        if "surf.k" in key:
            rd[key.replace("surf", "endo")] = value * 2.0

    rd["endo"], rd["activeEndo"], rd["sortF"], rd["kRec"], rd["kDeg"] = tuple(rxntfr[17:22])
    rd["Rexpr_2Ra"], rd["Rexpr_2Rb"], rd["Rexpr_gc"], rd["Rexpr_15Ra"], rd["Rexpr_7R"], rd["Rexpr_9R"], rd["Rexpr_4Ra"], rd["Rexpr_21Ra"] = tuple(rxntfr[22:30])

    return rd


def getRateVec(rxntfr):
    """Retrieves and unpacks ordered dict + constructs rate vector for model fitting"""
    entries = rxntfr.size
    rxnlength = rxntfr.shape[0]

    if (entries / rxnlength) > 1:
        FullRateVec = np.zeros([rxntfr.shape[0], 60])
        for row in range(rxntfr.shape[0]):
            ratesParamsDict = getparamsdict(rxntfr[row, :])
            FullRateVec[row, :] = np.array(list(ratesParamsDict.values()), dtype=np.float)
    else:
        FullRateVec = np.zeros(60)
        ratesParamsDict = getparamsdict(rxntfr)
        FullRateVec = np.array(list(ratesParamsDict.values()), dtype=np.float)

    return FullRateVec


mutaff = {"WT N-term": [0.19, 5.296], "WT C-term": [0.54, 3.043], "V91K C-term": [0.69, 7.5586], "R38Q N-term": [0.71, 3.9949], "F42Q N-Term": [9.48, 2.815], "N88D C-term": [1.01, 24.0166]}


def getMutAffDict():
    """Returns a dictionary containing mutant dissociation constants for 2Ra and BGc"""
    return mutaff


def mut_adjust(rxntfr, mutdict, mut_name):
    """Adjust alpha beta and gamma affinities for muteins prior to run through model according to BLI data"""
    # Adjust a affinities
    input_params = mutdict.get(mut_name)

    # change for unkVec instead of dict
    rxntfr[:, 7] = input_params[0] * 0.6
    rxntfr[:, 27] = rxntfr[:, 7] * 5.0

    # Adjust b/g affinities
    for jj in range(0, rxntfr.shape[0]):
        bg_adjust = (input_params[1] * 0.6) / rxntfr[jj, 10]
        for ii in [8, 9, 10, 11, 12, 28, 29, 30, 31, 32]:
            rxntfr[jj, ii] = rxntfr[jj, ii] * bg_adjust

    return rxntfr
