"""
This creates Figure S6. Full Partial Derivative Bar
"""
import string
import numpy as np
import theano.tensor as T
import theano
from .figureCommon import subplotLabel, getSetup, grouped_scaling
from ..model import getTotalActiveSpecies, receptor_expression, getRateVec, getparamsdict
from ..imports import import_Rexpr, import_samples_2_15, import_pstat
from ..differencing_op import runCkineDoseOp


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((30, 8), (2, 1))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    Specificity(ax=ax[0])
    Spec_Aff(ax[1], "T-reg", 40, Op, unkVecT, scalesT)

    return f


def Specificity(ax):
    """ Creates Theano Function for calculating Specificity gradient with respect to various parameters"""
    S_partials = Sfunc(unkVec.flatten()) / S.eval()
    S_partials = np.delete(S_partials, np.s_[-8:])
    S_partials = np.delete(S_partials, np.s_[7:21])
    S_partials = np.delete(S_partials, np.s_[13:27])
    y_pos = np.arange(S_partials.size)
    ax.set_ylim(-0.1, 0.1)
    Derivs = ax.bar(y_pos, S_partials, width=1, align='center', alpha=1)
    barlabel(Derivs, ax)


def OPgen(unkVecOP, CellTypes, OpC, scalesTh, RaAffM, RbAffM):
    "Generates the UnkVec with cell specific receptor abundances and expression rates"
    _, receptor_dataC, cell_names_receptorC = import_Rexpr()
    cell_names_receptorC = cell_names_receptorC.tolist()

    # set true Op
    cell_data = receptor_dataC[cell_names_receptorC.index(CellTypes), :]
    unkVecOP = T.set_subtensor(unkVecOP[46], receptor_expression(cell_data[0], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # RA
    unkVecOP = T.set_subtensor(unkVecOP[47], receptor_expression(cell_data[1], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Rb
    unkVecOP = T.set_subtensor(unkVecOP[48], receptor_expression(cell_data[2], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Gc
    unkVecOP = T.set_subtensor(unkVecOP[49], 0)
    unkVecOP = T.set_subtensor(unkVecOP[1], unkVecOP[1] * RaAffM)  # krev
    unkVecOP = T.set_subtensor(unkVecOP[21], unkVecOP[21] * RaAffM)
    unkVecOP = T.set_subtensor(unkVecOP[2], unkVecOP[2] * RbAffM)  # k2rev
    unkVecOP = T.set_subtensor(unkVecOP[22], unkVecOP[22] * RbAffM)
    unkVecOP = T.set_subtensor(unkVecOP[3], unkVecOP[3] * RbAffM)  # k4rev
    unkVecOP = T.set_subtensor(unkVecOP[24], unkVecOP[24] * RbAffM)
    unkVecOP = T.set_subtensor(unkVecOP[4], unkVecOP[4] * RbAffM)  # k5rev
    unkVecOP = T.set_subtensor(unkVecOP[24], unkVecOP[24] * RbAffM)
    unkVecOP = T.set_subtensor(unkVecOP[5], unkVecOP[5] * RbAffM)  # k10rev
    unkVecOP = T.set_subtensor(unkVecOP[25], unkVecOP[25] * RbAffM)
    unkVecOP = T.set_subtensor(unkVecOP[6], unkVecOP[6] * RbAffM)  # k11rev
    unkVecOP = T.set_subtensor(unkVecOP[26], unkVecOP[26] * RbAffM)

    cell_groups = np.array([['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+', 'Naive CD8+', 'Mem CD8+']])
    for i, group in enumerate(cell_groups):
        group = np.array(group)
        if np.where(group == CellTypes)[0].size > 0:
            scale1 = scalesTh[i, 1, 0]
            scale2 = scalesTh[i, 0, 0]

    Cell_Op = (OpC(unkVecOP) * scale1) / (OpC(unkVecOP) + scale2)

    return Cell_Op


def barlabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    ODunk = np.zeros(60)
    OD = getparamsdict(ODunk)
    labels = list(OD.keys())
    labels = np.array(labels)
    labels = np.delete(labels, np.s_[-8:])
    labels = np.delete(labels, np.s_[0:6])
    labels = np.delete(labels, np.s_[7:21])
    labels = np.delete(labels, np.s_[13:27])
    labels = labels.tolist()
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if height < 0 or height > 0.1:
            height = 0
        ax.annotate(labels[i],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(-5, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='left', va='bottom',
                    rotation=45)


def genscalesT(unkVecOP):
    """ This generates the group of scaling constants for our OPs """
    _, scale = import_samples_2_15(N=1)
    _, receptor_dataC, cell_names_receptorC = import_Rexpr()
    cell_names_receptorC = cell_names_receptorC.tolist()
    ckineConcT, _, IL2_data, IL15_data, _ = import_pstat()
    tps = np.array([0.5, 1., 2., 4.]) * 60
    Cond2 = np.zeros((1, 6), dtype=np.float64)
    Cond15 = np.zeros((1, 6), dtype=np.float64)
    pred2Vec, pred15Vec = np.zeros([len(cell_names_receptorC), ckineConcT.size, 1, len(tps)]), np.zeros([len(cell_names_receptorC), ckineConcT.size, 1, len(tps)])

    for i, Ctype in enumerate(cell_names_receptorC):  # Update each vec for unique cell expression levels
        cell_data = receptor_dataC[cell_names_receptorC.index(Ctype), :]
        unkVecOP = T.set_subtensor(unkVecOP[46], receptor_expression(cell_data[0], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # RA
        unkVecOP = T.set_subtensor(unkVecOP[47], receptor_expression(cell_data[1], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Rb
        unkVecOP = T.set_subtensor(unkVecOP[48], receptor_expression(cell_data[2], unkVecOP[41], unkVecOP[44], unkVecOP[43], unkVecOP[45]))  # Gc
        unkVecOP = T.set_subtensor(unkVecOP[49], 0)  # 15
        # unkVecOP = T.set_subtensor(unkVecOP[1], unkVecOP[1]*100)  # Zoe Use this to set affinities
        for j, conc in enumerate(ckineConcT):
            Cond2[0, 0] = conc
            Cond15[0, 1] = conc
            for k, timeT in enumerate(tps):
                # calculate full tensor of predictions for all cells for all timepoints
                ScaleOp = runCkineDoseOp(tt=np.array(timeT), condense=getTotalActiveSpecies().astype(np.float64), conditions=Cond2)
                pred2Vec[i, j, 0, k] = ScaleOp(unkVecOP).eval()
                ScaleOp = runCkineDoseOp(tt=np.array(timeT), condense=getTotalActiveSpecies().astype(np.float64), conditions=Cond15)
                pred15Vec[i, j, 0, k] = ScaleOp(unkVecOP).eval()

    scalesTh = grouped_scaling(scale, cell_names_receptorC, IL2_data, IL15_data, pred2Vec, pred15Vec)

    return scalesTh


def Spec_Aff(ax, cell, npoints, OpAff, unkVecAff, scalesAff):
    "Plots specificity for a cell type over a range of IL2RBG and IL2Ra affinities"
    affRange = np.logspace(2, -1, npoints)
    RaAff = np.array([1, 10])
    specHolder = np.zeros([len(RaAff), npoints])
    for i, k1Aff in enumerate(RaAff):
        for j, k5Aff in enumerate(affRange):
            Saff = (OPgen(unkVecAff, cell, OpAff, scalesAff, k1Aff, k5Aff) /
                    (OPgen(unkVecAff, "T-reg", OpAff, scalesAff, k1Aff, k5Aff) +
                     OPgen(unkVecAff, "T-helper", OpAff, scalesAff, k1Aff, k5Aff) +
                     OPgen(unkVecAff, "NK", OpAff, scalesAff, k1Aff, k5Aff) +
                     OPgen(unkVecAff, "CD8+", OpAff, scalesAff, k1Aff, k5Aff)))
            specHolder[i, j] = Saff.eval()
        ax.plot(1 / affRange, specHolder[i, :], label=str(1 / RaAff[i]) + " IL2Ra Affinity")

    ax.set_xscale('log')
    ax.set_xlabel('Relative CD122/CD132 Affinity')
    ax.set_ylabel('T-reg Specificity')
    ax.legend()


ckineConc, _, _, _, _ = import_pstat()
ckineC = ckineConc[7]
time = 240.
unkVec, scales = import_samples_2_15(N=1)
_, receptor_data, cell_names_receptor = import_Rexpr()
unkVec = getRateVec(unkVec)
CondIL = np.zeros((1, 6), dtype=np.float64)
CondIL[0, 0] = ckineC
Op = runCkineDoseOp(tt=np.array(time), condense=getTotalActiveSpecies().astype(np.float64), conditions=CondIL)
unkVecTrunc = T.zeros(54)
unkVec = unkVec[6::].flatten()
unkVecTrunc = T.set_subtensor(unkVecTrunc[0:], np.transpose(unkVec))
unkVecT = unkVecTrunc
scalesT = genscalesT(unkVecT)

S = (OPgen(unkVecT, "T-reg", Op, scalesT, 1, 1) /
     (OPgen(unkVecT, "T-reg", Op, scalesT, 1, 1) +
      OPgen(unkVecT, "T-helper", Op, scalesT, 1, 1) +
      OPgen(unkVecT, "NK", Op, scalesT, 1, 1) +
      OPgen(unkVecT, "CD8+", Op, scalesT, 1, 1)))

Sgrad = T.grad(S[0], unkVecT)
Sfunc = theano.function([unkVecT], Sgrad)
