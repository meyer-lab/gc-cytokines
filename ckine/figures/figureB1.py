"""
This creates Figure 1.
"""
import string
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..model import runCkineU_IL2, ligandDeg, getTotalActiveCytokine
from ..make_tensor import rxntfR


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    ax, f = getSetup((11, 6), (2, 2))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    halfMax_IL2RaAff(ax[0])
    activeReceptorComplexes(ax[1])
    halfMax_IL2RbAff(ax[2])
    halfMax_IL2RbAff_highIL2Ra(ax[3])

    return f


def dRespon(input_params, CD25=1.0):
    """ Calculate an IL2 dose response curve. """
    ILs = np.logspace(-3.0, 3.0)
    activee = np.array([runIL2simple(input_params, ii, CD25) for ii in ILs])

    return ILs, activee


def IC50global(x, y):
    """ Calculate half-maximal concentration w.r.t. wt. """
    return np.interp(20.0, y, x)


changesA = np.logspace(-1, 1.5, num=20)
changesB = np.array([0.0, 0.1, 0.25, 0.5, 1.0])
output = np.zeros((changesA.size, changesB.size))


def halfMax_IL2RaAff(ax):
    """ Plots half maximal IL2 concentration for varied IL2Ra and IL2Rb affinities. """
    changesA_a = np.logspace(-2, 1, num=20)
    changesB_a = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    for i, itemA in enumerate(changesA_a):
        for j, itemB in enumerate(changesB_a):
            ILs, BB = dRespon([itemA, itemB, 5.0])
            output[i, j] = IC50global(ILs, BB)
    for ii in range(output.shape[1]):
        ax.loglog(changesA_a, output[:, ii], label=str(changesB_a[ii]))
    ax.loglog([0.01, 10.], [0.17, 0.17], 'k-')
    ax.set(ylabel='Half-Maximal IL2 Concentration [nM]', xlabel='IL2Ra-IL2 Kd (fold wt)', ylim=(0.01, 20))
    ax.legend(title="IL2Rb Kd v wt")


def activeReceptorComplexes(ax):
    """ Plots active receptor complexes per cell across increasing IL2 concentration for wild type and adjusted affinities. """
    wt = dRespon([1.0, 1.0, 5.0])
    ten = dRespon([0.1, 5.0, 5.0])

    ax.semilogx(wt[0], wt[1], label="wt")
    ax.semilogx(ten[0], ten[1], 'r', label="10X higher/lower affinity IL2Ra/IL2Rb")
    ax.set(ylabel='Active Receptor Complexes (#/cell)', xlabel='IL2 [nM]')
    ax.legend()


def halfMax_IL2RbAff(ax):
    """ Plots half maximal IL2 concentration across decreasing IL2Rb affinity for varied IL2Ra expression levels using wild type IL2Ra affinity. """
    for i, itemA in enumerate(changesA):
        for j, itemB in enumerate(changesB):
            ILs, BB = dRespon([1.0, itemA, 5.0], CD25=itemB)
            output[i, j] = IC50global(ILs, BB)

    for ii in range(output.shape[1]):
        ax.loglog(changesA, output[:, ii], label=str(changesB[ii]))

    ax.loglog([0.1, 10.], [0.17, 0.17], 'k-')
    ax.set(ylabel='Half-Maximal IL2 Concentration [nM]', xlabel='IL2Rb-IL2 Kd (relative to wt)', ylim=(0.001, 10), xlim=(0.1, 10))
    ax.legend(title="CD25 rel expr")


def halfMax_IL2RbAff_highIL2Ra(ax):
    """ Plots half maximal IL2 concentration across decreasing IL2Rb affinity for varied IL2Ra expression levels using 10x
    increased IL2Ra affinity. """
    for i, itemA in enumerate(changesA):
        for j, itemB in enumerate(changesB):
            ILs, BB = dRespon([0.1, itemA, 5.0], CD25=itemB)
            output[i, j] = IC50global(ILs, BB)

    for ii in range(output.shape[1]):
        ax.loglog(changesA, output[:, ii], label=str(changesB[ii]))

    ax.loglog([0.1, 10.], [0.17, 0.17], 'k-')
    ax.set(ylabel='Half-Maximal IL2 Concentration [nM]', xlabel='IL2Rb-IL2 Kd (relative to wt)', ylim=(0.001, 10), xlim=(0.1, 10))
    ax.legend(title="CD25 rel expr")


def runIL2simple(input_params, IL, CD25=1.0, ligandDegradation=False):
    """ Version to focus on IL2Ra/Rb affinity adjustment. """
    tps = np.array([500.0])

    kfwd, k4rev, k5rev = rxntfR[6], rxntfR[7], rxntfR[8]

    k1rev = 0.6 * 10 * input_params[0]
    k2rev = 0.6 * 144 * input_params[1]
    k11rev = 63.0 * k5rev / 1.5 * input_params[1]
    IL2Ra, IL2Rb, gc = rxntfR[22] * CD25, rxntfR[23], rxntfR[24]

    # IL, kfwd, k1rev, k2rev, k4rev, k5rev, k11rev, R, R, R
    rxntfr = np.array([IL, kfwd, k1rev, k2rev, k4rev, k5rev, k11rev, IL2Ra, IL2Rb,
                       gc, k1rev * input_params[2], k2rev * input_params[2],
                       k4rev * input_params[2], k5rev * input_params[2], k11rev * input_params[2]])
    # input_params[2] represents endosomal binding affinity relative to surface affinity

    yOut, retVal = runCkineU_IL2(tps, rxntfr)

    assert retVal == 0

    if ligandDegradation:
        # rate of ligand degradation
        return ligandDeg(yOut[0], sortF=rxntfR[19], kDeg=rxntfR[21], cytokineIDX=0)

    return getTotalActiveCytokine(0, np.squeeze(yOut))
