"""
A file that includes the model and important helper functions.
"""
import os
import ctypes as ct
import numpy as np
import pymc3 as pm
import theano.tensor as T
from collections import OrderedDict



filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ckine.so")
libb = ct.cdll.LoadLibrary(filename)
pcd = ct.POINTER(ct.c_double)
libb.fullModel_C.argtypes = (pcd, ct.c_double, pcd, pcd)
libb.runCkine.argtypes = (pcd, ct.c_uint, pcd, pcd, ct.c_bool, ct.c_double, pcd)
libb.runCkineParallel.argtypes = (pcd, pcd, ct.c_uint, ct.c_uint, pcd, ct.c_double, pcd)
libb.runCkineSParallel.argtypes = (pcd, pcd, ct.c_uint, ct.c_uint, pcd, pcd, pcd, ct.c_double, pcd)

__nSpecies = 62


def nSpecies():
    """ Returns the total number of species in the model. """
    return __nSpecies


__halfL = 28


def halfL():
    """ Returns the number of species on the surface alone. """
    return __halfL


__nParams = 30
__rxParams = 60


def nParams():
    """ Returns the length of the rxntfR vector. """
    return __nParams

def rxParams():
    """ Returns the length of the rxntfR vector. """
    return __rxParams


__internalStrength = 0.5  # strength of endosomal activity relative to surface


def internalStrength():
    """Returns the internalStrength of endosomal activity."""
    return __internalStrength


__internalV = 623.0  # endosomal volume


def internalV():
    """ Returns __internalV. """
    return __internalV


__nRxn = 17


def nRxn():
    """ Returns the length of the rxn rates vector (doesn't include traf rates). """
    return __nRxn


def runCkineU_IL2(tps, rxntfr):
    """ Standard version of solver that returns species abundances given times and unknown rates. """
    rxntfr = rxntfr.copy()
    assert rxntfr.size == 15
    
    yOut = np.zeros((tps.size, __nSpecies), dtype=np.float64)
    rxntfr = getRateVec(rxntfr)

    retVal = libb.runCkine(tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size, yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)), True, 0.0, None)

    assert retVal >= 0  # make sure solver worked

    return yOut


def runCkineU(tps, rxntfr, preT=0.0, prestim=None):
    """ Standard version of solver that returns species abundances given times and unknown rates. """
    return runCkineUP(tps, np.atleast_2d(rxntfr.copy()), preT, prestim)


def runCkineUP(tps, rxntfr, preT=0.0, prestim=None):
    """ Version of runCkine that runs in parallel. """
    tps = np.array(tps)
    assert rxntfr.size % 30 == 0

    assert (rxntfr[:, 19] < 1.0).all()  # Check that sortF won't throw
    assert np.all(np.any(rxntfr > 0.0, axis=1))  # make sure at least one element is >0 for all rows
   

    yOut = np.zeros((rxntfr.shape[0] * tps.size, __nSpecies), dtype=np.float64)

    rxntfr = getRateVec(rxntfr)

    if preT != 0.0:
        assert preT > 0.0
        assert prestim.size == 6
        prestim = prestim.ctypes.data_as(ct.POINTER(ct.c_double))

    retVal = libb.runCkineParallel(
        rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)), tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size, rxntfr.shape[0], yOut.ctypes.data_as(ct.POINTER(ct.c_double)), preT, prestim
    )

    assert retVal >= 0  # make sure solver worked

    return yOut


def runCkineSP(tps, rxntfr, actV, preT=0.0, prestim=None):
    """ Version of runCkine that runs in parallel. """
    tps = np.array(tps)
    #assert rxntfr.size % __nParams == 0
    assert (rxntfr[:, 19] < 1.0).all()  # Check that sortF won't throw

    yOut = np.zeros((rxntfr.shape[0] * tps.size), dtype=np.float64)

    rxntfr = getRateVec(rxntfr)
    sensV = np.zeros((rxntfr.shape[0] * tps.size, __rxParams), dtype=np.float64, order="C")
    

    if preT != 0.0:
        assert preT > 0.0
        assert prestim.size == 6
        prestim = prestim.ctypes.data_as(ct.POINTER(ct.c_double))

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
    sensVnp = np.array([sensV])
    np.delete(sensVnp, [0, 6], 1)
    np.delete(sensVnp, [1, 3], 1)
    np.delete(sensVnp, [3, 7], 1)
    np.delete(sensVnp, [7], 1)
    np.delete(sensVnp, [8], 1)
    np.delete(sensVnp, [9], 1)
    np.delete(sensVnp, [10], 1)
    np.delete(sensVnp, [11, 31], 1)
    sensV = np.tolist(sensVnp)

    return (yOut, retVal, sensV)


def fullModel(y, t, rxntfr):
    """ Implement the full model based on dydt, trafficking, expression. """
    assert rxntfr.size == 30

    yOut = np.zeros_like(y)

    rxntfr = getRateVec(rxntfr)

    libb.fullModel_C(y.ctypes.data_as(ct.POINTER(ct.c_double)), t, yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)))

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
    return getActiveCytokine(cytokineIDX, yVec[0:__halfL]) + __internalStrength * getActiveCytokine(cytokineIDX, yVec[__halfL: __halfL * 2])


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
    return surfaceReceptors(yVec) + __internalStrength * surfaceReceptors(yVec[__halfL: __halfL * 2])


def ligandDeg(yVec, sortF, kDeg, cytokineIDX):
    """ This function calculates rate of total ligand degradation. """
    yVec_endo_species = yVec[__halfL: (__halfL * 2)].copy()  # get all endosomal complexes
    yVec_endo_lig = yVec[(__halfL * 2)::].copy()  # get all endosomal ligands
    sum_active = np.sum(getActiveCytokine(cytokineIDX, yVec_endo_species))
    __cytok_species_IDX = np.zeros(__halfL, dtype=np.bool)  # create array of size halfL
    __cytok_species_IDX[getCytokineSpecies()[cytokineIDX]] = 1  # assign 1's for species corresponding to the cytokineIDX
    sum_total = np.sum(yVec_endo_species * __cytok_species_IDX)
    sum_inactive = (sum_total - sum_active) * sortF  # scale the inactive species by sortF
    return kDeg * (((sum_inactive + sum_active) * __internalStrength) + (yVec_endo_lig[cytokineIDX] * __internalV))  # can assume all free ligand and active species are degraded at rate kDeg


def receptor_expression(receptor_abundance, endo, kRec, sortF, kDeg):
    """ Uses receptor abundance (from flow) and trafficking rates to calculate receptor expression rate at steady state. """
    rec_ex = (receptor_abundance * endo) / (1.0 + ((kRec * (1.0 - sortF)) / (kDeg * sortF)))
    return rec_ex


def getparamsdict(rxntfr):
    """Where rate vectors and constants are defined, organized in an ordered dictionary"""
    ratesParamsDict = OrderedDict()

    if rxntfr.size == 15:
        ratesParamsDict['Il2'] = rxntfr[0]
        ratesParamsDict['Il15'] = 0.0
        ratesParamsDict['Il7'] = 0.0
        ratesParamsDict['Il9'] = 0.0
        ratesParamsDict['Il4'] = 0.0
        ratesParamsDict['Il21'] = 0.0
        ratesParamsDict['kfbnd'] = float(0.60)
        ratesParamsDict['kfwd'] = rxntfr[1]
        ratesParamsDict['surface.k1rev'] = rxntfr[2]
        ratesParamsDict['surface.k2rev'] = rxntfr[3]
        ratesParamsDict['surface.k4rev'] = rxntfr[4]
        ratesParamsDict['surface.k5rev'] = rxntfr[5]
        ratesParamsDict['surface.k10rev'] = 12.0 * ratesParamsDict['surface.k5rev'] / 1.5
        ratesParamsDict['surface.k11rev'] = rxntfr[6]
        ratesParamsDict['surface.k13rev'] = 1.0
        ratesParamsDict['surface.k14rev'] = 1.0
        ratesParamsDict['surface.k16rev'] = 1.0
        ratesParamsDict['surface.k17rev'] = 1.0
        ratesParamsDict['surface.k22rev'] = 1.0
        ratesParamsDict['surface.k23rev'] = 1.0
        ratesParamsDict['surface.k25rev'] = 1.0
        ratesParamsDict['surface.k27rev'] = 1.0
        ratesParamsDict['surface.k29rev'] = 1.0
        ratesParamsDict['surface.k31rev'] = 1.0
        ratesParamsDict['surface.k32rev'] = 1.0
        ratesParamsDict['surface.k33rev'] = 1.0
        ratesParamsDict['surface.k34rev'] = 1.0
        ratesParamsDict['surface.k35rev'] = 1.0
        ratesParamsDict['endosome.k1rev'] = rxntfr[10]
        ratesParamsDict['endosome.k2rev'] = rxntfr[11]
        ratesParamsDict['endosome.k4rev'] = rxntfr[12]
        ratesParamsDict['endosome.k5rev'] = rxntfr[13]
        ratesParamsDict['endosome.k10rev'] = ratesParamsDict['surface.k10rev']
        ratesParamsDict['endosome.k11rev'] = rxntfr[14]
        ratesParamsDict['endosome.k13rev'] = ratesParamsDict['surface.k13rev']
        ratesParamsDict['endosome.k14rev'] = ratesParamsDict['surface.k14rev']
        ratesParamsDict['endosome.k16rev'] = ratesParamsDict['surface.k16rev']
        ratesParamsDict['endosome.k17rev'] = ratesParamsDict['surface.k17rev']
        ratesParamsDict['endosome.k22rev'] = ratesParamsDict['surface.k22rev']
        ratesParamsDict['endosome.k23rev'] = ratesParamsDict['surface.k23rev']
        ratesParamsDict['endosome.k25rev'] = ratesParamsDict['surface.k25rev']
        ratesParamsDict['endosome.k27rev'] = ratesParamsDict['surface.k27rev']
        ratesParamsDict['endosome.k29rev'] = ratesParamsDict['surface.k29rev']
        ratesParamsDict['endosome.k31rev'] = ratesParamsDict['surface.k31rev']
        ratesParamsDict['endosome.k32rev'] = ratesParamsDict['surface.k32rev']
        ratesParamsDict['endosome.k33rev'] = ratesParamsDict['surface.k33rev']
        ratesParamsDict['endosome.k34rev'] = ratesParamsDict['surface.k34rev']
        ratesParamsDict['endosome.k35rev'] = ratesParamsDict['surface.k35rev']
        ratesParamsDict['endo'] = 0.08221
        ratesParamsDict['activeEndo'] = 2.52654
        ratesParamsDict['sortF'] = 0.16024
        ratesParamsDict['kRec'] = 0.10017
        ratesParamsDict['kDeg'] = 0.00807
        ratesParamsDict['Rexpr_2Ra'] = rxntfr[7]
        ratesParamsDict['Rexpr_2Rb'] = rxntfr[8]
        ratesParamsDict['Rexpr_gc'] = rxntfr[9]
        ratesParamsDict['Rexpr_15Ra'] = 0.0
        ratesParamsDict['Rexpr_7R'] = 0.0
        ratesParamsDict['Rexpr_9R'] = 0.0
        ratesParamsDict['Rexpr_4Ra'] = 0.0
        ratesParamsDict['Rexpr_21Ra'] = 0.0
        
    else:
        ratesParamsDict['Il2'] = rxntfr[0]
        ratesParamsDict['Il15'] = rxntfr[1]
        ratesParamsDict['Il7'] = rxntfr[2]
        ratesParamsDict['Il9'] = rxntfr[3]
        ratesParamsDict['Il4'] = rxntfr[4]
        ratesParamsDict['Il21'] = rxntfr[5]
        ratesParamsDict['kfbnd'] = float(0.60)
        ratesParamsDict['surface.kfwd'] = rxntfr[6]
        ratesParamsDict['surface.k1rev'] = np.array([ratesParamsDict['kfbnd'] * 10.0])
        ratesParamsDict['surface.k2rev'] = np.array([ratesParamsDict['kfbnd'] * 144.0])
        ratesParamsDict['surface.k4rev'] = rxntfr[7]
        ratesParamsDict['surface.k5rev'] = rxntfr[8]
        ratesParamsDict['surface.k10rev'] = 12.0 * ratesParamsDict['surface.k5rev'] / 1.5
        ratesParamsDict['surface.k11rev'] = 63.0 * ratesParamsDict['surface.k5rev'] / 1.5
        ratesParamsDict['surface.k13rev'] = ratesParamsDict['kfbnd'] * 0.065
        ratesParamsDict['surface.k14rev'] = ratesParamsDict['kfbnd'] * 438.0
        ratesParamsDict['surface.k16rev'] = rxntfr[9]
        ratesParamsDict['surface.k17rev'] = rxntfr[10]
        ratesParamsDict['surface.k22rev'] = rxntfr[11]
        ratesParamsDict['surface.k23rev'] = rxntfr[12]
        ratesParamsDict['surface.k25rev'] = ratesParamsDict['kfbnd'] * 59.0
        ratesParamsDict['surface.k27rev'] = rxntfr[13]
        ratesParamsDict['surface.k29rev'] = ratesParamsDict['kfbnd'] * 0.1
        ratesParamsDict['surface.k31rev'] = rxntfr[14]
        ratesParamsDict['surface.k32rev'] = ratesParamsDict['kfbnd'] * 1.0
        ratesParamsDict['surface.k33rev'] = rxntfr[15]
        ratesParamsDict['surface.k34rev'] = ratesParamsDict['kfbnd'] * 0.07
        ratesParamsDict['surface.k35rev'] = rxntfr[16]
        ratesParamsDict['endosome.k1rev'] = ratesParamsDict['surface.k1rev'] * 5.0
        ratesParamsDict['endosome.k2rev'] = ratesParamsDict['surface.k2rev'] * 5.0
        ratesParamsDict['endosome.k4rev'] = ratesParamsDict['surface.k4rev'] * 5.0
        ratesParamsDict['endosome.k5rev'] = ratesParamsDict['surface.k5rev'] * 5.0
        ratesParamsDict['endosome.k10rev'] = ratesParamsDict['surface.k10rev'] * 5.0
        ratesParamsDict['endosome.k11rev'] = ratesParamsDict['surface.k11rev'] * 5.0
        ratesParamsDict['endosome.k13rev'] = ratesParamsDict['surface.k13rev'] * 5.0
        ratesParamsDict['endosome.k14rev'] = ratesParamsDict['surface.k14rev'] * 5.0
        ratesParamsDict['endosome.k16rev'] = ratesParamsDict['surface.k16rev'] * 5.0
        ratesParamsDict['endosome.k17rev'] = ratesParamsDict['surface.k17rev'] * 5.0
        ratesParamsDict['endosome.k22rev'] = ratesParamsDict['surface.k22rev'] * 5.0
        ratesParamsDict['endosome.k23rev'] = ratesParamsDict['surface.k23rev'] * 5.0
        ratesParamsDict['endosome.k25rev'] = ratesParamsDict['surface.k25rev'] * 5.0
        ratesParamsDict['endosome.k27rev'] = ratesParamsDict['surface.k27rev'] * 5.0
        ratesParamsDict['endosome.k29rev'] = ratesParamsDict['surface.k29rev'] * 5.0
        ratesParamsDict['endosome.k31rev'] = ratesParamsDict['surface.k31rev'] * 5.0
        ratesParamsDict['endosome.k32rev'] = ratesParamsDict['surface.k32rev'] * 5.0
        ratesParamsDict['endosome.k33rev'] = ratesParamsDict['surface.k33rev'] * 5.0
        ratesParamsDict['endosome.k34rev'] = ratesParamsDict['surface.k34rev'] * 5.0
        ratesParamsDict['endosome.k35rev'] = ratesParamsDict['surface.k35rev'] * 5.0
        ratesParamsDict['endo'] = rxntfr[17]
        ratesParamsDict['activeEndo'] = rxntfr[18]
        ratesParamsDict['sortF'] = rxntfr[19]
        ratesParamsDict['kRec'] = rxntfr[20]
        ratesParamsDict['kDeg'] = rxntfr[21]
        ratesParamsDict['Rexpr_2Ra'] = rxntfr[22]
        ratesParamsDict['Rexpr_2Rb'] = rxntfr[23]
        ratesParamsDict['Rexpr_gc'] = rxntfr[24]
        ratesParamsDict['Rexpr_15Ra'] = rxntfr[25]
        ratesParamsDict['Rexpr_7R'] = rxntfr[26]
        ratesParamsDict['Rexpr_9R'] = rxntfr[27]
        ratesParamsDict['Rexpr_4Ra'] = rxntfr[28]
        ratesParamsDict['Rexpr_21Ra'] = rxntfr[29]
        

    return ratesParamsDict


def getRateVec(rxntfr):
    """Retrieves and unpacks ordered dict + constructs rate vector for model fitting"""
    entries = rxntfr.size
    rxnlength = rxntfr.shape[0]
    
    if (entries/rxnlength) > 1:
        FullRateVec = np.zeros([rxntfr.shape[0], 60])
        numRuns = np.linspace(0, rxntfr.shape[0] - 1, rxntfr.shape[0])
        for i in numRuns:
            row = int(i)
            ratesParamsDict = getparamsdict(rxntfr[row, :])
            del ratesParamsDict['kfbnd']
            FullRateVec[row, :] = np.array(list(ratesParamsDict.values()))
            for i, rate in enumerate(FullRateVec[row, :]):
                if isinstance(rate, float) == False:
                    FullRateVec[row, i] = float(FullRateVec[row, i])

    else:
        FullRateVec = np.zeros(60)
        ratesParamsDict = getparamsdict(rxntfr)
        del ratesParamsDict['kfbnd']
        FullRateVec = np.array(list(ratesParamsDict.values()))
        for i, rate in enumerate(FullRateVec):
            if isinstance(rate, float) == False:
                FullRateVec[i] = float(FullRateVec[i])
        

    return FullRateVec
