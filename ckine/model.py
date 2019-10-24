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


def nParams():
    """ Returns the length of the rxntfR vector. """
    return __nParams


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

    retVal = libb.runCkine(tps.ctypes.data_as(ct.POINTER(ct.c_double)), tps.size, yOut.ctypes.data_as(ct.POINTER(ct.c_double)), rxntfr.ctypes.data_as(ct.POINTER(ct.c_double)), True, 0.0, None)

    assert retVal >= 0  # make sure solver worked

    return yOut


def runCkineU(tps, rxntfr, preT=0.0, prestim=None):
    """ Standard version of solver that returns species abundances given times and unknown rates. """
    return runCkineUP(tps, np.atleast_2d(rxntfr.copy()), preT, prestim)


def runCkineUP(tps, rxntfr, preT=0.0, prestim=None):
    """ Version of runCkine that runs in parallel. """
    tps = np.array(tps)
    assert rxntfr.size % __nParams == 0
    assert rxntfr.shape[1] == __nParams

    assert (rxntfr[:, 19] < 1.0).all()  # Check that sortF won't throw
    assert np.all(np.any(rxntfr > 0.0, axis=1))  # make sure at least one element is >0 for all rows

    yOut = np.zeros((rxntfr.shape[0] * tps.size, __nSpecies), dtype=np.float64)
    print("UPMod")
    print(rxntfr)
    trafvec = [rxntfr[:, 6], rxntfr[:, 17], rxntfr[:, 18], rxntfr[:, 21], rxntfr[:, 22], rxntfr[:, 20]]

    if rxntfr[:, 23] == pm.Lognormal('Rexpr_2Ra', sd=0.5, shape=1):
        IL2Ra = True
    else:
        IL2Ra = False

    _, rxntfr = getRateVec(trafvec, IL2Ra)

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
    assert rxntfr.size % __nParams == 0
    assert rxntfr.shape[1] == __nParams
    assert (rxntfr[:, 19] < 1.0).all()  # Check that sortF won't throw
    print("SPMod")
    print(rxntfr)

    yOut = np.zeros((rxntfr.shape[0] * tps.size), dtype=np.float64)
    sensV = np.zeros((rxntfr.shape[0] * tps.size, __nParams), dtype=np.float64, order="C")
    trafvec = [rxntfr[:, 6], rxntfr[:, 17], rxntfr[:, 18], rxntfr[:, 21], rxntfr[:, 22], rxntfr[:, 20]]


    if rxntfr[:, 23] == pm.Lognormal('Rexpr_2Ra', sd=0.5, shape=1):
        IL2Ra = True
    else:
        IL2Ra = False

    _, rxntfr = getRateVec(trafvec, IL2Ra)

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

    return (yOut, retVal, sensV)


def fullModel(y, t, rxntfr):
    """ Implement the full model based on dydt, trafficking, expression. """
    assert rxntfr.size == __nParams

    yOut = np.zeros_like(y)
    print("Full Model")
    print(rxntfr)

    trafvec = [rxntfr[6], rxntfr[17], rxntfr[18], rxntfr[21], rxntfr[22], rxntfr[20]]

    if rxntfr[23] == pm.Lognormal('Rexpr_2Ra', sd=0.5, shape=1):
        IL2Ra = True
    else:
        IL2Ra = False

    _, rxntfr = getRateVec(trafvec, IL2Ra)

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


def getparamsdict(trafvec, IL2Ra=True):
    """Where rate vectors and constants are defined, organized in an ordered dictionary"""
    ratesParamsDict = OrderedDict()
    kfwd, endo, activeEndo, kRec, kDeg, sortF = trafvec[0], trafvec[1], trafvec[2], trafvec[3], trafvec[4], trafvec[5]
    ratesParamsDict['kfbnd'] = float(0.60)
    if IL2Ra:
        ratesParamsDict['Rexpr_2Ra'] = pm.Lognormal('Rexpr_2Ra', sd=0.5, shape=1)
    else:
        ratesParamsDict['Rexpr_2Ra'] = T.zeros(1, dtype=np.float64)
    ratesParamsDict['Rexpr_15Ra'] = pm.Lognormal('Rexpr_15Ra', sd=0.5, shape=1)
    ratesParamsDict['Rexpr_2Rb'] = pm.Lognormal('Rexpr_2Rb', sd=0.5, shape=1)
    ratesParamsDict['Rexpr_gc'] = pm.Lognormal('Rexpr_gc', sd=0.5, shape=1)
    ratesParamsDict['kfwd'] = kfwd
    ratesParamsDict['endo'] = endo
    ratesParamsDict['activeEndo'] = activeEndo
    ratesParamsDict['kRec'] = kRec
    ratesParamsDict['kDeg'] = kDeg
    ratesParamsDict['sortF'] = sortF
    ratesParamsDict['surface.k1rev'] = np.array([ratesParamsDict['kfbnd'] * 10])
    ratesParamsDict['surface.k2rev'] = np.array([ratesParamsDict['kfbnd'] * 144])
    ratesParamsDict['surface.k4rev'] = pm.Lognormal('surface.k4rev', sd=0.5, shape=1)
    ratesParamsDict['surface.k5rev'] = pm.Lognormal('surface.k5rev', sd=0.5, shape=1)
    ratesParamsDict['surface.k10rev'] = 12.0 * ratesParamsDict['surface.k5rev'] / 1.5
    ratesParamsDict['surface.k11rev'] = 63.0 * ratesParamsDict['surface.k5rev'] / 1.5
    ratesParamsDict['surface.k13rev'] = np.array([ratesParamsDict['kfbnd'] * 0.065])
    ratesParamsDict['surface.k14rev'] = np.array([ratesParamsDict['kfbnd'] * 438])
    ratesParamsDict['surface.k16rev'] = pm.Lognormal('surface.k16rev', sd=0.5, shape=1)
    ratesParamsDict['surface.k17rev'] = pm.Lognormal('surface.k17rev', sd=0.5, shape=1)
    ratesParamsDict['surface.k22rev'] = pm.Lognormal('surface.k22rev', sd=0.5, shape=1)
    ratesParamsDict['surface.k23rev'] = pm.Lognormal('surface.k23rev', sd=0.5, shape=1)
    ratesParamsDict['surface.k25rev'] = np.array([ratesParamsDict['kfbnd'] * 59])
    ratesParamsDict['surface.k27rev'] = T.ones(1, dtype=np.float64)
    ratesParamsDict['surface.k29rev'] = np.array([ratesParamsDict['kfbnd'] * 0.1])
    ratesParamsDict['surface.k31rev'] = T.ones(1, dtype=np.float64)
    ratesParamsDict['surface.k32rev'] = np.array([ratesParamsDict['kfbnd'] * 1.0])
    ratesParamsDict['surface.k33rev'] = T.ones(1, dtype=np.float64)
    ratesParamsDict['surface.k34rev'] = np.array([ratesParamsDict['kfbnd'] * 0.07])
    ratesParamsDict['surface.k35rev'] = T.ones(1, dtype=np.float64)
    ratesParamsDict['endosome.k1rev'] = np.array([ratesParamsDict['kfbnd'] * 10 * 5.0])
    ratesParamsDict['endosome.k2rev'] = np.array([ratesParamsDict['kfbnd'] * 144 * 5.0])
    ratesParamsDict['endosome.k4rev'] = ratesParamsDict['surface.k4rev'] * 5.0
    ratesParamsDict['endosome.k5rev'] = ratesParamsDict['surface.k5rev'] * 5.0
    ratesParamsDict['endosome.k10rev'] = ratesParamsDict['surface.k10rev'] * 5.0
    ratesParamsDict['endosome.k11rev'] = ratesParamsDict['surface.k11rev'] * 5.0
    ratesParamsDict['endosome.k13rev'] = np.array([ratesParamsDict['kfbnd'] * 0.065 * 5.0])
    ratesParamsDict['endosome.k14rev'] = np.array([ratesParamsDict['kfbnd'] * 438 * 5.0])
    ratesParamsDict['endosome.k16rev'] = ratesParamsDict['surface.k16rev'] * 5.0
    ratesParamsDict['endosome.k17rev'] = ratesParamsDict['surface.k17rev'] * 5.0
    ratesParamsDict['endosome.k22rev'] = ratesParamsDict['surface.k22rev'] * 5.0
    ratesParamsDict['endosome.k23rev'] = ratesParamsDict['surface.k23rev'] * 5.0
    ratesParamsDict['endosome.k25rev'] = np.array([ratesParamsDict['kfbnd'] * 59 * 5.0])
    ratesParamsDict['endosome.k27rev'] = ratesParamsDict['surface.k27rev'] * 5.0
    ratesParamsDict['endosome.k29rev'] = np.array([ratesParamsDict['kfbnd'] * 0.1 * 5.0])
    ratesParamsDict['endosome.k31rev'] = ratesParamsDict['surface.k31rev'] * 5.0
    ratesParamsDict['endosome.k32rev'] = np.array([ratesParamsDict['kfbnd'] * 1.0 * 5.0])
    ratesParamsDict['endosome.k33rev'] = ratesParamsDict['surface.k33rev'] * 5.0
    ratesParamsDict['endosome.k34rev'] = np.array([ratesParamsDict['kfbnd'] * 0.07 * 5.0])
    ratesParamsDict['endosome.k35rev'] = ratesParamsDict['surface.k35rev'] * 5.0
    ratesParamsDict['Null Rates1'] = T.ones(1, dtype=np.float64)
    ratesParamsDict['Null Rates2'] = T.ones(1, dtype=np.float64)
    ratesParamsDict['Null Rates3'] = T.ones(1, dtype=np.float64)
    ratesParamsDict['Null Rates4'] = T.ones(1, dtype=np.float64)

    return ratesParamsDict


def getRateVec(trafvec, IL2Ra=True):
    """Retrieves and unpacks ordered dict + constructs rate vector for model fitting"""

    ratesParamsDict = getparamsdict(trafvec, IL2Ra)


    FullRateVec = T.concatenate(
        (ratesParamsDict['kfwd'],
         T.as_tensor_variable(ratesParamsDict['surface.k1rev']),
         T.as_tensor_variable(ratesParamsDict['surface.k2rev']),
         ratesParamsDict['surface.k4rev'],
         ratesParamsDict['surface.k5rev'],
         ratesParamsDict['surface.k10rev'],
         ratesParamsDict['surface.k11rev'],
         T.as_tensor_variable(ratesParamsDict['surface.k13rev']),
         T.as_tensor_variable(ratesParamsDict['surface.k14rev']),
         ratesParamsDict['surface.k16rev'],
         ratesParamsDict['surface.k17rev'],
         ratesParamsDict['surface.k22rev'],
         ratesParamsDict['surface.k23rev'],
         T.as_tensor_variable(ratesParamsDict['surface.k25rev']),
         ratesParamsDict['surface.k27rev'],
         T.as_tensor_variable(ratesParamsDict['surface.k29rev']),
         ratesParamsDict['surface.k31rev'],
         T.as_tensor_variable(ratesParamsDict['surface.k32rev']),
         ratesParamsDict['surface.k33rev'],
         T.as_tensor_variable(ratesParamsDict['surface.k34rev']),
         ratesParamsDict['surface.k35rev'],
         T.as_tensor_variable(ratesParamsDict['endosome.k1rev']),
         T.as_tensor_variable(ratesParamsDict['endosome.k2rev']),
         ratesParamsDict['endosome.k4rev'],
         ratesParamsDict['endosome.k5rev'],
         ratesParamsDict['endosome.k10rev'],
         ratesParamsDict['endosome.k11rev'],
         T.as_tensor_variable(ratesParamsDict['endosome.k13rev']),
         T.as_tensor_variable(ratesParamsDict['endosome.k14rev']),
         ratesParamsDict['endosome.k16rev'],
         ratesParamsDict['endosome.k17rev'],
         ratesParamsDict['endosome.k22rev'],
         ratesParamsDict['endosome.k23rev'],
         T.as_tensor_variable(ratesParamsDict['endosome.k25rev']),
         ratesParamsDict['endosome.k27rev'],
         T.as_tensor_variable(ratesParamsDict['endosome.k29rev']),
         ratesParamsDict['endosome.k31rev'],
         T.as_tensor_variable(ratesParamsDict['endosome.k32rev']),
         ratesParamsDict['endosome.k33rev'],
         T.as_tensor_variable(ratesParamsDict['endosome.k34rev']),
         ratesParamsDict['endosome.k35rev'],
         ratesParamsDict['endo'],
         ratesParamsDict['activeEndo'],
         ratesParamsDict['sortF'],
         ratesParamsDict['kRec'],
         ratesParamsDict['kDeg'],
         ratesParamsDict['Rexpr_2Ra'],
         ratesParamsDict['Rexpr_2Rb'],
         ratesParamsDict['Rexpr_gc'],
         ratesParamsDict['Rexpr_15Ra'],
         ratesParamsDict['Null Rates1'] * 0.0,
         ratesParamsDict['Null Rates2'] * 0.0,
         ratesParamsDict['Null Rates3'] * 0.0,
         ratesParamsDict['Null Rates4'] * 0.0), axis = 0)

    unkVec = T.concatenate(
        (ratesParamsDict['kfwd'],
         ratesParamsDict['surface.k4rev'],
         ratesParamsDict['surface.k5rev'],
         ratesParamsDict['surface.k16rev'],
         ratesParamsDict['surface.k17rev'],
         ratesParamsDict['surface.k22rev'],
         ratesParamsDict['surface.k23rev'],
         ratesParamsDict['surface.k27rev'],
         ratesParamsDict['surface.k31rev'],
         ratesParamsDict['surface.k33rev'],
         ratesParamsDict['surface.k35rev'],
         ratesParamsDict['endo'],
         ratesParamsDict['activeEndo'],
         ratesParamsDict['sortF'],
         ratesParamsDict['kRec'],
         ratesParamsDict['kDeg'],
         ratesParamsDict['Rexpr_2Ra'],
         ratesParamsDict['Rexpr_2Rb'],
         ratesParamsDict['Rexpr_gc'],
         ratesParamsDict['Rexpr_15Ra'],
         ratesParamsDict['Null Rates1'],
         ratesParamsDict['Null Rates2'],
         ratesParamsDict['Null Rates3'],
         ratesParamsDict['Null Rates4']))

    return unkVec, FullRateVec
