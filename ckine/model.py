import numpy as np
from scipy.integrate import odeint
import copy

try:
    from numba import jit, float64, boolean as numbabool
except ImportError:
    print('Numba not available, so no JIT compile.')

    def jit(ob):
        return ob

    float64 = None
    numbabool = None


__active_species_IDX = np.zeros(26, dtype=np.bool)
__active_species_IDX[np.array([8, 9, 16, 17, 21, 25])] = 1


@jit(float64[26](float64[26], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64), nopython=True, cache=True, nogil=True)
def dy_dt(y, t, IL2, IL15, IL7, IL9, kfwd, k5rev, k6rev, k15rev, k17rev, k18rev, k22rev, k23rev, k26rev, k27rev, k29rev, k30rev, k31rev):
    # IL2 in nM
    IL2Ra, IL2Rb, gc, IL2_IL2Ra, IL2_IL2Rb, IL2_gc, IL2_IL2Ra_IL2Rb, IL2_IL2Ra_gc, IL2_IL2Rb_gc, IL2_IL2Ra_IL2Rb_gc = y[0:10]
    
    # IL15 in nM
    IL15Ra, IL15_IL15Ra, IL15_IL2Rb, IL15_gc, IL15_IL15Ra_IL2Rb, IL15_IL15Ra_gc, IL15_IL2Rb_gc, IL15_IL15Ra_IL2Rb_gc = y[10:18]
    
    #IL7, IL9 in nM
    IL7Ra, IL7Ra_IL7, gc_IL7, IL7Ra_gc_IL7, IL9R, IL9R_IL9, gc_IL9, IL9R_gc_IL9 = y[18:26] # k25 - k32

    # These are probably measured in the literature
    kfbnd = 0.01 # Assuming on rate of 10^7 M-1 sec-1
    k1rev = kfbnd * 10 # doi:10.1016/j.jmb.2004.04.038, 10 nM

    k2rev = kfbnd * 144 # doi:10.1016/j.jmb.2004.04.038, 144 nM
    k3fwd = kfbnd / 10.0 # Very weak, > 50 uM. Voss, et al (1993). PNAS. 90, 2428–2432.
    k3rev = 50000 * k3fwd
    k10rev = 12.0 * k5rev / 1.5 # doi:10.1016/j.jmb.2004.04.038
    k11rev = 63.0 * k5rev / 1.5 # doi:10.1016/j.jmb.2004.04.038
    
    # Literature values for k values for IL-15
    k13rev = kfbnd * 0.065 #based on the multiple papers suggesting 30-100 pM
    k14rev = kfbnd * 438 # doi:10.1038/ni.2449, 438 nM
    
    # Literature values for IL-7
    k25rev = kfbnd * 59. # DOI:10.1111/j.1600-065X.2012.01160.x, 59 nM
    
    # To satisfy detailed balance these relationships should hold
    # _Based on initial assembly steps
    k4rev = kfbnd * k6rev * k3rev / k1rev / k3fwd
    k7rev = k3fwd * k2rev * k5rev / kfbnd / k3rev
    k12rev = k1rev * k11rev / k2rev
    # _Based on formation of full complex
    k9rev = k2rev * k10rev * k12rev / kfbnd / k3rev / k6rev * k3fwd
    k8rev = k2rev * k10rev * k12rev / kfbnd / k7rev / k3rev * k3fwd

    # IL15
    # To satisfy detailed balance these relationships should hold
    # _Based on initial assembly steps
    k16rev = kfwd * k18rev * k15rev / k13rev / kfbnd
    k19rev = kfwd * k14rev * k17rev / kfbnd / k15rev
    k24rev = k13rev * k23rev / k14rev
    # _Based on formation of full complex

    k21rev = k14rev * k22rev * k24rev / kfwd / k15rev / k18rev * kfbnd
    k20rev = k14rev * k22rev * k24rev / k19rev / k15rev

    # _One detailed balance IL7/9 loop
    k32rev = k29rev * k31rev / k30rev
    k28rev = k25rev * k27rev / k26rev

    dydt = y.copy()
    
    # IL2
    dydt[0] = -kfbnd * IL2Ra * IL2 + k1rev * IL2_IL2Ra - kfwd * IL2Ra * IL2_gc + k6rev * IL2_IL2Ra_gc - kfwd * IL2Ra * IL2_IL2Rb_gc + k8rev * IL2_IL2Ra_IL2Rb_gc - kfwd * IL2Ra * IL2_IL2Rb + k12rev * IL2_IL2Ra_IL2Rb
    dydt[1] = -kfbnd * IL2Rb * IL2 + k2rev * IL2_IL2Rb - kfwd * IL2Rb * IL2_gc + k7rev * IL2_IL2Rb_gc - kfwd * IL2Rb * IL2_IL2Ra_gc + k9rev * IL2_IL2Ra_IL2Rb_gc - kfwd * IL2Rb * IL2_IL2Ra + k11rev * IL2_IL2Ra_IL2Rb
    dydt[2] = -k3fwd * IL2 * gc + k3rev * IL2_gc - kfwd * IL2_IL2Rb * gc + k5rev * IL2_IL2Rb_gc - kfwd * IL2_IL2Ra * gc + k4rev * IL2_IL2Ra_gc - kfwd * IL2_IL2Ra_IL2Rb * gc + k10rev * IL2_IL2Ra_IL2Rb_gc
    dydt[3] = -kfwd * IL2_IL2Ra * IL2Rb + k11rev * IL2_IL2Ra_IL2Rb - kfwd * IL2_IL2Ra * gc + k4rev * IL2_IL2Ra_gc + kfbnd * IL2 * IL2Ra - k1rev * IL2_IL2Ra
    dydt[4] = -kfwd * IL2_IL2Rb * IL2Ra + k12rev * IL2_IL2Ra_IL2Rb - kfwd * IL2_IL2Rb * gc + k5rev * IL2_IL2Rb_gc + kfbnd * IL2 * IL2Rb - k2rev * IL2_IL2Rb
    dydt[5] = -kfwd *IL2_gc * IL2Ra + k6rev * IL2_IL2Ra_gc - kfwd * IL2_gc * IL2Rb + k7rev * IL2_IL2Rb_gc + k3fwd * IL2 * gc - k3rev * IL2_gc
    dydt[6] = -kfwd * IL2_IL2Ra_IL2Rb * gc + k10rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra * IL2Rb - k11rev * IL2_IL2Ra_IL2Rb + kfwd * IL2_IL2Rb * IL2Ra - k12rev * IL2_IL2Ra_IL2Rb
    dydt[7] = -kfwd * IL2_IL2Ra_gc * IL2Rb + k9rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra * gc - k4rev * IL2_IL2Ra_gc + kfwd * IL2_gc * IL2Ra - k6rev * IL2_IL2Ra_gc
    dydt[8] = -kfwd * IL2_IL2Rb_gc * IL2Ra + k8rev * IL2_IL2Ra_IL2Rb_gc + kfwd * gc * IL2_IL2Rb - k5rev * IL2_IL2Rb_gc + kfwd * IL2_gc * IL2Rb - k7rev * IL2_IL2Rb_gc
    dydt[9] = kfwd * IL2_IL2Rb_gc * IL2Ra - k8rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra_gc * IL2Rb - k9rev * IL2_IL2Ra_IL2Rb_gc + kfwd * IL2_IL2Ra_IL2Rb * gc - k10rev * IL2_IL2Ra_IL2Rb_gc

    # IL15
    dydt[10] = -kfbnd * IL15Ra * IL15 + k13rev * IL15_IL15Ra - kfbnd * IL15Ra * IL15_gc + k18rev * IL15_IL15Ra_gc - kfwd * IL15Ra * IL15_IL2Rb_gc + k20rev * IL15_IL15Ra_IL2Rb_gc - kfwd * IL15Ra * IL15_IL2Rb + k24rev * IL15_IL15Ra_IL2Rb
    dydt[11] = -kfwd * IL15_IL15Ra * IL2Rb + k23rev * IL15_IL15Ra_IL2Rb - kfwd * IL15_IL15Ra * gc + k16rev * IL15_IL15Ra_gc + kfbnd * IL15 * IL15Ra - k13rev * IL15_IL15Ra
    dydt[12] = -kfwd * IL15_IL2Rb * IL15Ra + k24rev * IL15_IL15Ra_IL2Rb - kfbnd * IL15_IL2Rb * gc + k17rev * IL15_IL2Rb_gc + kfbnd * IL15 * IL2Rb - k14rev * IL15_IL2Rb 
    dydt[13] = -kfbnd * IL15_gc * IL15Ra + k18rev * IL15_IL15Ra_gc - kfwd * IL15_gc * IL2Rb + k19rev * IL15_IL2Rb_gc + kfbnd * IL15 * gc - k15rev * IL15_gc
    dydt[14] = -kfwd * IL15_IL15Ra_IL2Rb * gc + k22rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra * IL2Rb - k23rev * IL15_IL15Ra_IL2Rb + kfwd * IL15_IL2Rb * IL15Ra - k24rev * IL15_IL15Ra_IL2Rb   
    dydt[15] = -kfwd * IL15_IL15Ra_gc * IL2Rb + k21rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra * gc - k16rev * IL15_IL15Ra_gc + kfbnd * IL15_gc * IL15Ra - k18rev * IL15_IL15Ra_gc
    dydt[16] = -kfwd * IL15_IL2Rb_gc * IL15Ra + k20rev * IL15_IL15Ra_IL2Rb_gc + kfbnd * gc * IL15_IL2Rb - k17rev * IL15_IL2Rb_gc + kfwd * IL15_gc * IL2Rb - k19rev * IL15_IL2Rb_gc
    dydt[17] =  kfwd * IL15_IL2Rb_gc * IL15Ra - k20rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra_gc * IL2Rb - k21rev * IL15_IL15Ra_IL2Rb_gc + kfwd * IL15_IL15Ra_IL2Rb * gc - k22rev * IL15_IL15Ra_IL2Rb_gc
    
    dydt[1] = dydt[1] - kfbnd * IL2Rb * IL15 + k14rev * IL15_IL2Rb - kfwd * IL2Rb * IL15_gc + k19rev * IL15_IL2Rb_gc - kfwd * IL2Rb * IL15_IL15Ra_gc + k21rev * IL15_IL15Ra_IL2Rb_gc - kfwd * IL2Rb * IL15_IL15Ra + k23rev * IL15_IL15Ra_IL2Rb
    dydt[2] = dydt[2] - kfbnd * IL15 * gc + k15rev * IL15_gc - kfbnd * IL15_IL2Rb * gc + k17rev * IL15_IL2Rb_gc - kfwd * IL15_IL15Ra * gc + k16rev * IL15_IL15Ra_gc - kfwd * IL15_IL15Ra_IL2Rb * gc + k22rev * IL15_IL15Ra_IL2Rb_gc
    
    # IL7
    dydt[2] = dydt[2] - kfbnd * IL7 * gc + k26rev * gc_IL7 - kfwd * gc * IL7Ra_IL7 + k27rev * IL7Ra_gc_IL7
    dydt[18] = -kfbnd * IL7Ra * IL7 + k25rev * IL7Ra_IL7 - kfwd * IL7Ra * gc_IL7 + k28rev * IL7Ra_gc_IL7
    dydt[19] = kfbnd * IL7Ra * IL7 - k25rev * IL7Ra_IL7 - kfwd * gc * IL7Ra_IL7 + k27rev * IL7Ra_gc_IL7
    dydt[20] = -kfwd * IL7Ra * gc_IL7 + k28rev * IL7Ra_gc_IL7 + kfbnd * IL7 * gc - k26rev * gc_IL7
    dydt[21] = kfwd * IL7Ra * gc_IL7 - k28rev * IL7Ra_gc_IL7 + kfwd * gc * IL7Ra_IL7 - k27rev * IL7Ra_gc_IL7

    # IL9
    dydt[2] = dydt[2] - kfbnd * IL9 * gc + k30rev * gc_IL9 - kfwd * gc * IL9R_IL9 + k31rev * IL9R_gc_IL9
    dydt[22] = -kfbnd * IL9R * IL9 + k29rev * IL9R_IL9 - kfwd * IL9R * gc_IL9 + k32rev * IL9R_gc_IL9
    dydt[23] = kfbnd * IL9R * IL9 - k29rev * IL9R_IL9 - kfwd * gc * IL9R_IL9 + k31rev * IL9R_gc_IL9
    dydt[24] = -kfwd * IL9R * gc_IL9 + k32rev * IL9R_gc_IL9 + kfbnd * IL9 * gc - k30rev * gc_IL9
    dydt[25] = kfwd * IL9R * gc_IL9 - k32rev * IL9R_gc_IL9 + kfwd * gc * IL9R_IL9 - k31rev * IL9R_gc_IL9

    return dydt


@jit(float64[4](float64[26]), nopython=True, cache=True, nogil=True)
def findLigConsume(dydt):
    """ Calculate the ligand consumption. """
    internalV = 623.0 # Same as that used in TAM model

    return -np.array([np.sum(dydt[3:10]), np.sum(dydt[11:18]), np.sum(dydt[19:22]), np.sum(dydt[23:26])]) / internalV


@jit(float64[52](float64[52], numbabool[26], float64, float64, float64, float64, float64, float64, float64[6]), nopython=True, cache=True, nogil=True)
def trafficking(y, activeV, endo, activeEndo, sortF, activeSortF, kRec, kDeg, exprV):
    """Implement trafficking."""

    dydt = np.empty_like(y)

    halfLen = len(y) // 2

    # Reconstruct a vector of active and inactive trafficking for vectorization
    endoV = np.full_like(activeV, endo)
    sortV = np.full_like(activeV, sortF)

    endoV[activeV == 1] = activeEndo
    sortV[activeV == 1] = activeSortF

    internalFrac = 0.5 # Same as that used in TAM model

    # Actually calculate the trafficking
    dydt[0:halfLen] = -y[0:halfLen]*endoV + kRec*(1-sortV)*y[halfLen::]*internalFrac # Endocytosis, recycling
    dydt[halfLen::] = y[0:halfLen]*endoV/internalFrac - kRec*(1-sortV)*y[halfLen::] - kDeg*sortV*y[halfLen::] # Endocytosis, recycling, degradation

    # Expression
    dydt[0:3] = dydt[0:3] + exprV[0:3] # IL2Ra, IL2Rb, gc
    dydt[10] = dydt[10] + exprV[3] # IL15Ra
    dydt[18] = dydt[18] + exprV[4] # IL7Ra
    dydt[22] = dydt[22] + exprV[5] # IL9R

    return dydt


def fullModel(y, t, r, trafRates):
    """Implement full model."""

    # Initialize vector
    dydt = np.empty_like(y)

    rxnL = 26

    # Calculate cell surface reactions
    dydt[0:rxnL] = dy_dt(y[0:rxnL], t, **r)

    # Calculate endosomal reactions
    dydt[rxnL:rxnL*2] = dy_dt(y[rxnL:rxnL*2], t, y[rxnL*2], y[rxnL*2+1], y[rxnL*2+2], y[rxnL*2+3],
                              r['kfwd'], r['k5rev'], r['k6rev'], r['k15rev'], r['k17rev'],
                              r['k18rev'], r['k22rev'], r['k23rev'], r['k26rev'],
                              r['k27rev'], r['k29rev'], r['k30rev'], r['k31rev'])

    # Handle trafficking
    # _Leave off the ligands on the end
    dydt[0:rxnL*2] = dydt[0:rxnL*2] + trafficking(y[0:rxnL*2], getActiveSpecies(), **trafRates)

    # Handle endosomal ligand balance.
    dydt[rxnL*2::] = findLigConsume(dydt[rxnL:rxnL*2])

    return dydt


def solveAutocrine(rxnRates, trafRates):
    rxnRates = copy.deepcopy(rxnRates)
    autocrineT = np.array([0.0, 100000.0])

    y0 = np.zeros(26*2 + 4, np.float64)

    # For now assume 0 autocrine ligand
    # TODO: Consider handling autocrine ligand more gracefully
    rxnRates['IL2'] = rxnRates['IL15'] = rxnRates['IL9'] = rxnRates['IL7'] = 0.0

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
